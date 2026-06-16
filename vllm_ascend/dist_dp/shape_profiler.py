"""
vLLM Ascend 算子 Shape 统计工具

使用方式:
    # 方式 1: 代码中使用 context manager (单卡)
    from shape_profiler import shape_profile
    with shape_profile("./output"):
        llm.generate(prompts)

    # 方式 2: 离线推理
    python shape_profiler.py --model Qwen/Qwen2-7B --prompt 'Hello'

    # 方式 3: Serve 模式 (单进程)
    python shape_profiler.py --model Qwen/Qwen2-7B --serve --port 8000

    # 方式 4: DP Serve 模式 (使用模板脚本, 类似 launch_online_dp.py)
    python shape_profiler.py --model /path/to/model --serve \\
        --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 \\
        --dp-address 141.61.52.167 --dp-rpc-port 12321 --vllm-start-port 8000 \\
        --dp-template ./run_dp_template.sh --output-dir ./output

    # 方式 5: 手动 DP 模式 (配合 launch_online_dp.py)
    export SHAPE_PROFILER_OUTPUT_DIR=./output
    python launch_online_dp.py --dp-size 32 --tp-size 1 ...
    # 运行完毕后合并报告:
    python shape_profiler.py --merge-only --output-dir ./output
"""

import atexit
import functools
import json
import multiprocessing
import os
import subprocess
import sys
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class OpOverloadPacketWrapper:
    """
    代理类，用于包装 torch.ops.* 的 OpOverloadPacket 对象。
    保留原始对象的所有属性（如 default, overloadpacket 等），
    同时在调用时记录 shape 信息。
    """
    
    def __init__(self, original_packet, op_name: str, profiler: "ShapeProfiler"):
        object.__setattr__(self, "_original", original_packet)
        object.__setattr__(self, "_op_name", op_name)
        object.__setattr__(self, "_profiler", profiler)
        object.__setattr__(self, "_shape_profiler_wrapped", True)
    
    def __call__(self, *args, **kwargs):
        result = self._original(*args, **kwargs)
        self._profiler.record_call(self._op_name, args, kwargs, result)
        return result
    
    def __getattr__(self, name):
        return getattr(self._original, name)
    
    def __getitem__(self, key):
        return self._original[key]
    
    def __repr__(self):
        return f"OpOverloadPacketWrapper({self._original})"
    
    def __str__(self):
        return str(self._original)


class TritonKernelWrapper:
    """
    代理类，用于包装 Triton kernel。
    支持 kernel[grid](...) 调用语法。
    """
    
    def __init__(self, original_kernel, op_name: str, profiler: "ShapeProfiler"):
        object.__setattr__(self, "_original", original_kernel)
        object.__setattr__(self, "_op_name", op_name)
        object.__setattr__(self, "_profiler", profiler)
        object.__setattr__(self, "_shape_profiler_wrapped", True)
    
    def __call__(self, *args, **kwargs):
        result = self._original(*args, **kwargs)
        self._profiler.record_call(self._op_name, args, kwargs, result)
        return result
    
    def __getitem__(self, key):
        original_item = self._original[key]
        return _TritonLaunchedKernelWrapper(original_item, self._op_name, self._profiler, key)
    
    def __getattr__(self, name):
        return getattr(self._original, name)
    
    def __repr__(self):
        return f"TritonKernelWrapper({self._original})"
    
    def __str__(self):
        return str(self._original)


class _TritonLaunchedKernelWrapper:
    """
    包装 Triton kernel[grid] 返回的对象。
    """
    
    def __init__(self, original_launched, op_name: str, profiler: "ShapeProfiler", grid_key):
        object.__setattr__(self, "_original", original_launched)
        object.__setattr__(self, "_op_name", op_name)
        object.__setattr__(self, "_profiler", profiler)
        object.__setattr__(self, "_grid_key", grid_key)
        object.__setattr__(self, "_shape_profiler_wrapped", True)
    
    def __call__(self, *args, **kwargs):
        result = self._original(*args, **kwargs)
        grid_info = str(self._grid_key) if self._grid_key else "unknown"
        self._profiler.record_call(f"{self._op_name}[{grid_info}]", args, kwargs, result)
        return result
    
    def __getattr__(self, name):
        return getattr(self._original, name)
    
    def __repr__(self):
        return f"_TritonLaunchedKernelWrapper({self._original})"


@dataclass
class ArgShapeInfo:
    shape: List[int]
    dtype: str
    count: int = 1


@dataclass
class OpShapeRecord:
    op_name: str
    call_count: int = 0
    # key: canonical call signature string -> count
    call_signatures: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "op_name": self.op_name,
            "call_count": self.call_count,
            "call_signatures": self.call_signatures,
        }


class ShapeProfiler:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ShapeProfiler._initialized:
            return
        ShapeProfiler._initialized = True

        self.original_funcs: Dict[str, Tuple[Any, Any]] = {}
        self.enabled = False
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.records: List[Dict] = []
        self._pid = os.getpid()
        self._record_file: Optional[str] = None
        self._record_file_handle = None
        self._is_single_card: bool = True
        self._record_count: int = 0
        atexit.register(self._cleanup)

    def _get_shape_dtype(self, obj: Any) -> List[Tuple[List[int], str]]:
        results = []
        if isinstance(obj, torch.Tensor):
            results.append((list(obj.shape), str(obj.dtype)))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                results.extend(self._get_shape_dtype(item))
        elif isinstance(obj, dict):
            for v in obj.values():
                results.extend(self._get_shape_dtype(v))
        return results

    def _store_record(self, record: Dict):
        if self._is_single_card and self._record_file_handle:
            try:
                self._record_file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._record_file_handle.flush()
                self._record_count += 1
                if self._record_count % 1000 == 0:
                    print(f"[ShapeProfiler] 已记录 {self._record_count} 条调用", flush=True)
            except Exception as e:
                print(f"[ShapeProfiler] 写入记录失败: {e}", flush=True)
        else:
            self.records.append(record)

    def save_records_to_file(self, output_file: str):
        if not self.records:
            return
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                for record in self.records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            print(f"[ShapeProfiler] 已保存 {len(self.records)} 条记录到 {output_file}", flush=True)
        except Exception as e:
            print(f"[ShapeProfiler] 保存记录失败: {e}", flush=True)

    def record_call(self, op_name: str, args: tuple, kwargs: dict, result: Any):
        if not self.enabled:
            return
        
        # call_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        arg_shapes = []
        for i, arg in enumerate(args):
            shapes_dtypes = self._get_shape_dtype(arg)
            if shapes_dtypes:
                for shape, dtype in shapes_dtypes:
                    arg_shapes.append({
                        "arg_idx": i,
                        "shape": shape,
                        "dtype": dtype
                    })
        
        kwarg_shapes = []
        for k, v in kwargs.items():
            shapes_dtypes = self._get_shape_dtype(v)
            if shapes_dtypes:
                for shape, dtype in shapes_dtypes:
                    kwarg_shapes.append({
                        "key": k,
                        "shape": shape,
                        "dtype": dtype
                    })
        
        output_shapes = []
        for shape, dtype in self._get_shape_dtype(result):
            output_shapes.append({"shape": shape, "dtype": dtype})
        
        record = {
            "op_name": op_name,
            "arg_shapes": arg_shapes,
            "kwarg_shapes": kwarg_shapes,
            "output_shapes": output_shapes,
            # "call_time": call_time,
            "pid": self._pid,
        }
        
        self._store_record(record)

    def _wrap_function(self, func: Callable, op_name: str) -> Callable:
        profiler_self = self
        call_count = [0]
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                print(f"[ShapeProfiler] DEBUG: 调用 {op_name} (第{call_count[0]}次), args数量={len(args)}, kwargs={list(kwargs.keys())}", flush=True)
            result = func(*args, **kwargs)
            profiler_self.record_call(op_name, args, kwargs, result)
            return result
        wrapped._shape_profiler_wrapped = True
        return wrapped

    def _install_module_hook(self, module_path: str, attr_name: str, op_name: str):
        try:
            import importlib
            module = importlib.import_module(module_path)
            
            original_func = getattr(module, attr_name)
            if getattr(original_func, "_shape_profiler_wrapped", False):
                print(f"[ShapeProfiler] DEBUG: 跳过已包装的 {op_name}", flush=True)
                return
            
            if hasattr(original_func, "__getitem__") and callable(original_func):
                wrapped = TritonKernelWrapper(original_func, op_name, self)
            else:
                wrapped = self._wrap_function(original_func, op_name)
            
            setattr(module, attr_name, wrapped)
            self.original_funcs[op_name] = (module, original_func)
            print(f"[ShapeProfiler] DEBUG: 成功安装 module hook: {module_path}.{attr_name} -> {op_name}", flush=True)
        except ImportError as e:
            print(f"[ShapeProfiler] DEBUG: 安装 module hook 失败 (ImportError): {module_path}.{attr_name} -> {e}", flush=True)
        except AttributeError as e:
            if "circular import" in str(e).lower():
                print(f"[ShapeProfiler] DEBUG: 安装 module hook 失败 (循环导入): {module_path}.{attr_name}", flush=True)
            else:
                print(f"[ShapeProfiler] DEBUG: 安装 module hook 失败 (AttributeError): {module_path}.{attr_name} -> {e}", flush=True)

    def _install_class_method_hook(self, module_path: str, class_name: str, method_name: str, op_name: str):
        try:
            import importlib
            module = importlib.import_module(module_path)
            
            cls = getattr(module, class_name)
            original_method = getattr(cls, method_name)
            
            if getattr(original_method, "_shape_profiler_wrapped", False):
                print(f"[ShapeProfiler] DEBUG: 跳过已包装的 {op_name}", flush=True)
                return
            
            wrapped = self._wrap_function(original_method, op_name)
            setattr(cls, method_name, wrapped)
            self.original_funcs[op_name] = (cls, original_method)
            print(f"[ShapeProfiler] DEBUG: 成功安装 class method hook: {module_path}.{class_name}.{method_name} -> {op_name}", flush=True)
        except ImportError as e:
            print(f"[ShapeProfiler] DEBUG: 安装 class method hook 失败 (ImportError): {module_path}.{class_name}.{method_name} -> {e}", flush=True)
        except AttributeError as e:
            print(f"[ShapeProfiler] DEBUG: 安装 class method hook 失败 (AttributeError): {module_path}.{class_name}.{method_name} -> {e}", flush=True)

    def _install_torch_op_hook(self, op_path: str, op_name: str):
        try:
            parts = op_path.split(".")
            if parts[0] == "torch":
                root = torch
            elif parts[0] == "torch_npu":
                import torch_npu
                root = torch_npu
            else:
                print(f"[ShapeProfiler] DEBUG: 跳过未知的 root: {parts[0]}", flush=True)
                return
            
            obj = root
            for part in parts[1:]:
                obj = getattr(obj, part)
            
            if getattr(obj, "_shape_profiler_wrapped", False):
                print(f"[ShapeProfiler] DEBUG: 跳过已包装的 {op_name}", flush=True)
                return
            
            if op_path.startswith("torch.ops."):
                wrapped = OpOverloadPacketWrapper(obj, op_name, self)
            else:
                wrapped = self._wrap_function(obj, op_name)
            
            parent = root
            for part in parts[1:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], wrapped)
            self.original_funcs[op_name] = (parent, obj)
            print(f"[ShapeProfiler] DEBUG: 成功安装 torch op hook: {op_path} -> {op_name}", flush=True)
        except AttributeError as e:
            print(f"[ShapeProfiler] DEBUG: 安装 torch op hook 失败 (AttributeError): {op_path} -> {e}", flush=True)
        except RuntimeError as e:
            print(f"[ShapeProfiler] DEBUG: 安装 torch op hook 失败 (RuntimeError): {op_path} -> {e}", flush=True)
        except ImportError as e:
            print(f"[ShapeProfiler] DEBUG: 安装 torch op hook 失败 (ImportError): {op_path} -> {e}", flush=True)

    def _hook_direct_register_custom_op(self):
        """
        Hook vllm.utils.torch_utils.direct_register_custom_op 函数，
        在注册自定义算子时自动包装 op_func。
        """
        try:
            from vllm.utils.torch_utils import direct_register_custom_op as original_register
            
            profiler = self
            
            def wrapped_register(op_name, op_func, mutates_args=None, fake_impl=None, target_lib=None, dispatch_key=None, tags=()):
                wrapped_func = profiler._wrap_function(op_func, f"custom_op.{op_name}")
                print(f"[ShapeProfiler] DEBUG: 包装 custom op: {op_name}", flush=True)
                return original_register(op_name, wrapped_func, mutates_args, fake_impl, target_lib, dispatch_key, tags)
            
            import vllm.utils.torch_utils
            vllm.utils.torch_utils.direct_register_custom_op = wrapped_register
            self.original_funcs["direct_register_custom_op"] = (vllm.utils.torch_utils, original_register)
            print(f"[ShapeProfiler] DEBUG: 成功 hook direct_register_custom_op", flush=True)
        except ImportError as e:
            print(f"[ShapeProfiler] DEBUG: hook direct_register_custom_op 失败 (ImportError): {e}", flush=True)
        except Exception as e:
            print(f"[ShapeProfiler] DEBUG: hook direct_register_custom_op 失败: {e}", flush=True)

    def install_hooks(self, output_dir: str = None, is_single_card: bool = True):
        if self.enabled:
            return

        self.output_dir = output_dir
        self.enabled = True
        self._is_single_card = is_single_card
        self._record_count = 0
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if self._is_single_card and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._record_file = os.path.join(output_dir, f"shape_records_pid{self._pid}.jsonl")
            try:
                self._record_file_handle = open(self._record_file, "w", encoding="utf-8")
                print(f"[ShapeProfiler] 单卡模式: 记录文件 {self._record_file}", flush=True)
            except Exception as e:
                print(f"[ShapeProfiler] 无法打开记录文件: {e}", flush=True)
                self._record_file_handle = None
        
        self._hook_direct_register_custom_op()

        triton_ops = [
            ("vllm_ascend.ops.triton.rope", "rope_forward_triton", "rope_forward_triton"),
            ("vllm_ascend.ops.triton.rope", "rope_forward_triton_siso", "rope_forward_triton_siso"),
            ("vllm.model_executor.layers.mamba.ops.causal_conv1d", "causal_conv1d_fn", "causal_conv1d_fn"),
            ("vllm_ascend.ops.triton.mamba.causal_conv1d", "causal_conv1d_update_npu", "causal_conv1d_update_npu"),
            ("vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope", "split_qkv_rmsnorm_rope_impl", "split_qkv_rmsnorm_rope_impl"),
            ("vllm_ascend.ops.triton.reject_sample", "rejection_greedy_sample_with_triton", "rejection_greedy_sample_with_triton"),
            ("vllm.v1.sample.rejection_sampler", "sample_recovered_tokens_kernel", "sample_recovered_tokens_kernel"),
            ("vllm.v1.sample.rejection_sampler", "expand_kernel", "expand_kernel"),
            ("vllm.v1.sample.rejection_sampler", "rejection_sample", "rejection_sample"),
            ("vllm_ascend.ops.triton.muls_add", "muls_add_triton", "muls_add"),
            ("vllm_ascend.ops.triton.fla.chunk", "chunk_gated_delta_rule", "chunk_gated_delta_rule"),
            ("vllm.model_executor.layers.fla.ops", "fused_recurrent_gated_delta_rule", "fused_recurrent_gated_delta_rule"),
            ("vllm_ascend.ops.triton.fla.l2norm", "l2norm_fwd", "l2norm_fwd_kernel"),
            ("vllm.model_executor.layers.layernorm", "fused_add_rms_norm", "fused_add_rms_norm"),
        ]

        for module_path, attr_name, op_name in triton_ops:
            self._install_module_hook(module_path, attr_name, op_name)

        # class_ops = [
        #     ("vllm_ascend.ops.layernorm", "AscendRMSNorm", "forward_oot", "AscendRMSNorm.forward_oot"),
        # ]

        torch_ops = [
            ("torch.ops._C_ascend.batch_matmul_transpose", "batch_matmul_transpose"),
            ("torch.ops._C_ascend.npu_add_rms_norm_bias", "npu_add_rms_norm_bias"),
            ("torch.ops._C_ascend.npu_gemma_rms_norm", "npu_gemma_rms_norm"),
            ("torch.ops._C_ascend.npu_sparse_flash_attention", "npu_sparse_flash_attention"),
            ("torch.ops._C_ascend.mla_preprocess", "mla_preprocess"),
            ("torch.ops._C_ascend.npu_lightning_indexer", "npu_lightning_indexer"),
            ("torch.ops._C_ascend.npu_lightning_indexer_quant", "npu_lightning_indexer_quant"),
            ("torch.ops._C_ascend.moe_gating_top_k", "moe_gating_top_k"),
            ("torch.ops._C_ascend.npu_moe_init_routing_custom", "npu_moe_init_routing_custom"),
            ("torch.ops._C_ascend.matmul_allreduce_add_rmsnorm", "matmul_allreduce_add_rmsnorm"),
            ("torch.ops._C_ascend.bgmv_shrink", "bgmv_shrink"),
            ("torch.ops._C_ascend.bgmv_expand", "bgmv_expand"),
            ("torch.ops._C_ascend.sgmv_shrink", "sgmv_shrink"),
            ("torch.ops._C_ascend.sgmv_expand", "sgmv_expand"),
            ("torch.ops._C_ascend.npu_apply_top_k_top_p", "npu_apply_top_k_top_p"),
            ("torch.ops._C_ascend.npu_causal_conv1d_custom", "npu_causal_conv1d_custom"),
            ("torch.ops.vllm.all_reduce", "all_reduce"),
            ("torch.ops.vllm.all_gather", "all_gather"),
            # ("torch.ops.vllm.maybe_all_gather_and_maybe_unpad", "maybe_all_gather_and_maybe_unpad"),
            ("torch.ops.vllm.muls_add", "muls_add"),
            ("torch.ops.mie_ops.npu_mla_preprocess", "npu_mla_preprocess"),
            ("torch_npu.npu_rms_norm", "npu_rms_norm"),
            ("torch_npu.npu_apply_rotary_pos_emb", "npu_apply_rotary_pos_emb"),
            ("torch_npu.npu_quantize", "npu_quantize"),
            ("torch_npu.npu_dynamic_quant", "npu_dynamic_quant"),
            ("torch_npu.npu_dynamic_mx_quant", "npu_dynamic_mx_quant"),
            ("torch_npu.npu_add_rms_norm_quant", "npu_add_rms_norm_quant"),
            ("torch_npu.npu_add_rms_norm_dynamic_quant", "npu_add_rms_norm_dynamic_quant"),
            ("torch_npu.npu_add_rms_norm", "npu_add_rms_norm"),
            ("torch_npu.npu_swiglu", "npu_swiglu"),
            ("torch_npu.npu_mm_reduce_scatter_base", "npu_mm_reduce_scatter_base"),
            ("torch_npu.npu_quant_matmul", "npu_quant_matmul"),
            ("torch.distributed.all_gather", "all_gather"),
            ("torch.index_select", "index_select"),
            ("torch.unique_consecutive", "unique_consecutive"),
            ("torch.nn.functional.linear", "linear"),
            ("torch.remainder", "remainder"),
            ("torch.split", "split"),
            ("torch.add", "add"),
            ("torch.max", "max"),
            ("torch.sub", "sub"),
            ("torch.rsub", "rsub"),
            ("torch.bmm", "bmm"),
            ("torch.transpose", "transpose"),
            ("torch.cat", "cat"),
            ("torch.cos", "cos"),
            ("torch.sin", "sin"),
            ("torch.nn.functional.silu", "silu"),
            ("torch.nn.functional.pad", "pad"),
            ("torch.sigmoid", "sigmoid"),
            ("torch.reshape", "reshape"),
            ("torch.distributed.all_gather_into_tensor", "all_gather_into_tensor"),
            ("torch.distributed.reduce_scatter_tensor", "reduce_scatter_tensor"),
            ("torch.distributed.all_to_all_single", "all_to_all_single"),
            ("torch_npu.npu_fused_infer_attention_score", "npu_fused_infer_attention_score"),
        ]

        for op_path, op_name in torch_ops:
            self._install_torch_op_hook(op_path, op_name)

        print(f"[ShapeProfiler] 已安装 {len(self.original_funcs)} 个算子 hook")

    def uninstall_hooks(self):
        for op_name, (parent, original_func) in self.original_funcs.items():
            if hasattr(original_func, "__name__"):
                setattr(parent, original_func.__name__, original_func)

        self.original_funcs.clear()
        self.enabled = False
        self.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if self._record_file_handle:
            try:
                self._record_file_handle.close()
                print(f"[ShapeProfiler] 已关闭记录文件，共 {self._record_count} 条记录", flush=True)
            except Exception as e:
                print(f"[ShapeProfiler] 关闭记录文件失败: {e}", flush=True)
            finally:
                self._record_file_handle = None

    def _cleanup(self):
        if self.enabled:
            self.uninstall_hooks()
        if self.records and self.output_dir and not self._is_single_card:
            output_file = os.path.join(self.output_dir, f"shape_records_pid{self._pid}.jsonl")
            self.save_records_to_file(output_file)

    def load_records(self, filepath: str) -> List[Dict]:
        if not filepath or not os.path.exists(filepath):
            return []
        
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def aggregate_records(self, records: List[Dict]) -> Dict[str, OpShapeRecord]:
        aggregated: Dict[str, OpShapeRecord] = {}
        
        for rec in records:
            op_name = rec["op_name"]
            
            if op_name not in aggregated:
                aggregated[op_name] = OpShapeRecord(
                    op_name=op_name,
                    # first_call_time=rec["call_time"],
                    # last_call_time=rec["call_time"],
                )
            
            aggregated[op_name].call_count += 1

            def _tensors_str(info):
                tensors = info.get("tensors") or ([{"shape": info["shape"], "dtype": info["dtype"]}] if "shape" in info else [])
                return "|".join(f"{t['shape']}:{t['dtype']}" for t in tensors)

            sig_parts = []
            for arg_info in rec.get("arg_shapes", []):
                sig_parts.append(f"arg{arg_info['arg_idx']}={_tensors_str(arg_info)}")
            for kwarg_info in rec.get("kwarg_shapes", []):
                sig_parts.append(f"{kwarg_info['key']}={_tensors_str(kwarg_info)}")
            for out_info in rec.get("output_shapes", []):
                sig_parts.append(f"out={out_info['shape']}:{out_info['dtype']}")
            sig = ", ".join(sig_parts)
            aggregated[op_name].call_signatures[sig] = aggregated[op_name].call_signatures.get(sig, 0) + 1
        
        return aggregated

    def export_json(self, filepath: str, records_file: str):
        records = self.load_records(records_file)
        if not records:
            print("[ShapeProfiler] 没有找到任何记录")
            return

        aggregated = self.aggregate_records(records)

        # 检测 DP 信息
        dp_ranks = set()
        for rec in records:
            if "dp_rank" in rec:
                dp_ranks.add(rec["dp_rank"])

        metadata = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_records": len(records),
            "unique_ops": len(aggregated),
        }
        if dp_ranks:
            metadata["dp_ranks"] = sorted(dp_ranks)
            metadata["dp_rank_count"] = len(dp_ranks)

        data = {
            "metadata": metadata,
            "records": [r.to_dict() for r in aggregated.values()],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[ShapeProfiler] JSON 已保存: {filepath}")

    def export_markdown(self, filepath: str, records_file: str):
        records = self.load_records(records_file)
        if not records:
            return

        aggregated = self.aggregate_records(records)

        # 检测 DP 信息
        dp_ranks = set()
        for rec in records:
            if "dp_rank" in rec:
                dp_ranks.add(rec["dp_rank"])

        lines = []
        lines.append("# vLLM Ascend 算子 Shape 统计报告\n\n")
        lines.append(f"**统计时间**: {self.start_time} - {self.end_time}\n\n")
        lines.append(f"**总记录数**: {len(records)}\n\n")
        lines.append(f"**唯一算子数**: {len(aggregated)}\n\n")
        if dp_ranks:
            lines.append(f"**DP Ranks**: {sorted(dp_ranks)} (共 {len(dp_ranks)} 个)\n\n")
        
        lines.append("## 算子调用统计\n\n")
        lines.append("| 算子名称 | 调用次数 | 唯一调用签名数 |\n")
        lines.append("|:---|:---:|:---:|\n")

        for op_name, rec in sorted(aggregated.items(), key=lambda x: -x[1].call_count):
            lines.append(f"| {op_name} | {rec.call_count} | {len(rec.call_signatures)} |\n")

        lines.append("\n## 详细调用签名分布\n\n")

        for op_name, rec in sorted(aggregated.items(), key=lambda x: -x[1].call_count):
            lines.append(f"### {op_name}\n\n")
            lines.append(f"- **调用次数**: {rec.call_count}\n")
            if rec.call_signatures:
                lines.append("\n#### 调用签名（入参+出参组合）\n\n")
                lines.append("| 签名 | 次数 |\n")
                lines.append("|:---|:---:|\n")
                for sig, count in sorted(rec.call_signatures.items(), key=lambda x: -x[1]):
                    lines.append(f"| `{sig}` | {count} |\n")
                lines.append("\n")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[ShapeProfiler] Markdown 已保存: {filepath}")

    def _merge_pid_files(self, output_dir: str) -> str:
        import glob

        all_records = []
        # 匹配所有记录文件: shape_records_pid*.jsonl 和 shape_records_dp*_pid*.jsonl
        patterns = [
            os.path.join(output_dir, "shape_records_pid*.jsonl"),
            os.path.join(output_dir, "shape_records_dp*_pid*.jsonl"),
        ]
        pid_files = set()
        for pattern in patterns:
            pid_files.update(glob.glob(pattern))
        pid_files = sorted(pid_files)

        for pid_file in pid_files:
            pid_records = self.load_records(pid_file)
            all_records.extend(pid_records)
            print(f"[ShapeProfiler] 合并 {os.path.basename(pid_file)}: {len(pid_records)} 条", flush=True)

        merged_file = os.path.join(output_dir, "shape_records_merged.jsonl")
        if all_records:
            with open(merged_file, "w", encoding="utf-8") as f:
                for record in all_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[ShapeProfiler] 合并后总记录数: {len(all_records)} 条", flush=True)

        return merged_file

    def generate_reports(self, output_dir: str, merge_pids: bool = False):
        os.makedirs(output_dir, exist_ok=True)
        
        records_file = None
        if merge_pids:
            records_file = self._merge_pid_files(output_dir)
        else:
            if self._is_single_card and self._record_file:
                records_file = self._record_file
                print(f"[ShapeProfiler] 从文件读取记录: {records_file}", flush=True)
            elif self.records:
                records_file = os.path.join(output_dir, f"shape_records_pid{self._pid}.jsonl")
                self.save_records_to_file(records_file)
        
        if records_file:
            self.export_json(os.path.join(output_dir, "shape_stats.json"), records_file)
            self.export_markdown(os.path.join(output_dir, "shape_stats_report.md"), records_file)


_profiler = ShapeProfiler()


def _install_worker_hooks(output_dir: str):
    _profiler.install_hooks(output_dir, is_single_card=True)


@contextmanager
def shape_profile(output_dir: str = "."):
    """
    上下文管理器，用于统计推理过程中的算子 shape

    使用示例:
        with shape_profile("./output"):
            llm.generate(prompts)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    os.environ["SHAPE_PROFILER_OUTPUT_DIR"] = output_dir
    
    _profiler.install_hooks(output_dir, is_single_card=True)
    
    try:
        yield _profiler
    finally:
        _profiler.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _profiler.uninstall_hooks()
        _profiler.generate_reports(output_dir)
        if "SHAPE_PROFILER_OUTPUT_DIR" in os.environ:
            del os.environ["SHAPE_PROFILER_OUTPUT_DIR"]


def get_profiler() -> ShapeProfiler:
    return _profiler


def enable_profiling(output_dir: str = ".", is_single_card: bool = True):
    _profiler.install_hooks(output_dir, is_single_card=is_single_card)


def disable_profiling():
    _profiler.uninstall_hooks()


if __name__ == "__main__":
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="vLLM Ascend 算子 Shape 统计工具")
    parser.add_argument("--model", type=str, help="模型名称或路径")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="输入提示")
    parser.add_argument("--max-tokens", type=int, default=100, help="最大生成 token 数")
    parser.add_argument("--output-dir", type=str, default=".", help="输出目录")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU 显存使用率")
    parser.add_argument("--max-model-len", type=int, default=None, help="最大序列长度")
    parser.add_argument("--max-num-seqs", type=int, default=16, help="最大并发序列数")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="张量并行大小")
    parser.add_argument("--serve", action="store_true", help="启动 serve 模式")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8010, help="服务端口")

    # DP 模式参数
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel 大小 (总 DP 数)")
    parser.add_argument("--dp-size-local", type=int, default=-1,
                        help="本地 DP 大小 (本机启动的 DP 进程数，默认等于 dp-size)")
    parser.add_argument("--dp-rank-start", type=int, default=0, help="本机 DP rank 起始值")
    parser.add_argument("--dp-address", type=str, default="", help="DP master 节点 IP 地址")
    parser.add_argument("--dp-rpc-port", type=str, default="12321", help="DP master 节点 RPC 端口")
    parser.add_argument("--vllm-start-port", type=int, default=8000,
                        help="DP 模式下各 rank 的 vLLM 端口起始值")
    parser.add_argument("--dp-template", type=str, default=None,
                        help="DP 模式模板脚本路径 (run_dp_template.sh)")

    # 报告合并模式
    parser.add_argument("--merge-only", action="store_true",
                        help="仅合并已有的记录文件并生成报告 (不启动推理)")

    args = parser.parse_args()

    # ==================== 仅合并报告模式 ====================
    if args.merge_only:
        if not os.path.isdir(args.output_dir):
            print(f"[ShapeProfiler] 错误: 输出目录不存在: {args.output_dir}")
            sys.exit(1)

        import glob
        record_files = glob.glob(os.path.join(args.output_dir, "shape_records_pid*.jsonl"))
        record_files.extend(glob.glob(os.path.join(args.output_dir, "shape_records_dp*_pid*.jsonl")))
        if not record_files:
            print(f"[ShapeProfiler] 错误: 在 {args.output_dir} 中未找到记录文件")
            sys.exit(1)

        print(f"[ShapeProfiler] 合并模式: 在 {args.output_dir} 中找到 {len(record_files)} 个记录文件")
        _profiler.start_time = "N/A"
        _profiler.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _profiler.generate_reports(args.output_dir, merge_pids=True)
        print(f"[ShapeProfiler] 报告生成完成")
        sys.exit(0)

    # ==================== 推理模式 (需要 --model) ====================
    if args.model:
        os.makedirs(args.output_dir, exist_ok=True)

        import glob
        old_files = glob.glob(os.path.join(args.output_dir, "shape_records_pid*.jsonl"))
        old_files.extend(glob.glob(os.path.join(args.output_dir, "shape_records_dp*_pid*.jsonl")))
        old_files.extend(glob.glob(os.path.join(args.output_dir, "shape_records_merged.jsonl")))
        for old_file in old_files:
            try:
                os.remove(old_file)
                print(f"[ShapeProfiler] 清理旧文件: {old_file}")
            except Exception:
                pass

        os.environ["SHAPE_PROFILER_OUTPUT_DIR"] = os.path.abspath(args.output_dir)

        if args.tensor_parallel_size > 1:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        _profiler.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ==================== DP Serve 模式 ====================
        if args.serve and args.dp_size > 1:
            dp_size = args.dp_size
            tp_size = args.tensor_parallel_size
            dp_size_local = args.dp_size_local if args.dp_size_local > 0 else dp_size
            dp_rank_start = args.dp_rank_start
            dp_address = args.dp_address
            dp_rpc_port = args.dp_rpc_port
            vllm_start_port = args.vllm_start_port
            dp_template = args.dp_template
            output_dir_abs = os.path.abspath(args.output_dir)

            if not dp_address:
                print("[ShapeProfiler] 错误: DP 模式需要指定 --dp-address")
                sys.exit(1)

            print(f"[ShapeProfiler] ========== DP Serve 模式 ==========", flush=True)
            print(f"[ShapeProfiler] 模型: {args.model}", flush=True)
            print(f"[ShapeProfiler] dp_size={dp_size}, tp_size={tp_size}, "
                  f"dp_size_local={dp_size_local}, dp_rank_start={dp_rank_start}", flush=True)
            print(f"[ShapeProfiler] dp_address={dp_address}, dp_rpc_port={dp_rpc_port}", flush=True)
            print(f"[ShapeProfiler] vllm_start_port={vllm_start_port}", flush=True)
            print(f"[ShapeProfiler] 输出目录: {output_dir_abs}", flush=True)

            if dp_template:
                # ============ 模板脚本模式 ============
                if not os.path.exists(dp_template):
                    print(f"[ShapeProfiler] 错误: 模板脚本不存在: {dp_template}")
                    sys.exit(1)

                print(f"[ShapeProfiler] 使用模板脚本: {dp_template}", flush=True)
                print(f"[ShapeProfiler] 按 Ctrl+C 停止所有 DP worker 并生成报告", flush=True)

                def _run_dp_worker_template(visible_devices, dp_rank, vllm_engine_port,
                                            dp_size, dp_address, dp_rpc_port, tp_size,
                                            dp_template_path, output_dir):
                    """在子进程中运行 DP worker (模板脚本模式)"""
                    os.environ["SHAPE_PROFILER_OUTPUT_DIR"] = output_dir
                    os.environ["SHAPE_PROFILER_DP_RANK"] = str(dp_rank)
                    os.environ["SHAPE_PROFILER_DP_SIZE"] = str(dp_size)

                    command = [
                        "bash",
                        dp_template_path,
                        visible_devices,
                        str(vllm_engine_port),
                        str(dp_size),
                        str(dp_rank),
                        dp_address,
                        dp_rpc_port,
                        str(tp_size),
                    ]
                    print(f"[ShapeProfiler] DP rank {dp_rank}: 启动命令 {' '.join(command)}",
                          flush=True)
                    subprocess.run(command, check=True)

                processes = []
                for i in range(dp_size_local):
                    dp_rank = dp_rank_start + i
                    vllm_engine_port = vllm_start_port + i
                    visible_devices = ",".join(
                        str(x) for x in range(i * tp_size, (i + 1) * tp_size))

                    p = multiprocessing.Process(
                        target=_run_dp_worker_template,
                        args=(visible_devices, dp_rank, vllm_engine_port,
                              dp_size, dp_address, dp_rpc_port, tp_size,
                              dp_template, output_dir_abs),
                    )
                    processes.append((dp_rank, p))
                    p.start()
                    print(f"[ShapeProfiler] DP rank {dp_rank}: 进程已启动 "
                          f"(pid={p.pid}, devices={visible_devices}, port={vllm_engine_port})",
                          flush=True)

                try:
                    for dp_rank, p in processes:
                        p.join()
                except KeyboardInterrupt:
                    print(f"\n[ShapeProfiler] 收到中断信号，正在停止所有 DP worker...",
                          flush=True)
                    for dp_rank, p in processes:
                        if p.is_alive():
                            p.terminate()
                    for dp_rank, p in processes:
                        p.join(timeout=15)
                        if p.is_alive():
                            p.kill()
                            p.join()
                        print(f"[ShapeProfiler] DP rank {dp_rank}: 进程已停止", flush=True)
                finally:
                    import time
                    time.sleep(2)
                    _profiler.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    _profiler.generate_reports(output_dir_abs, merge_pids=True)
                    if "SHAPE_PROFILER_OUTPUT_DIR" in os.environ:
                        del os.environ["SHAPE_PROFILER_OUTPUT_DIR"]

            else:
                # ============ 直接启动 vllm serve 模式 ============
                print(f"[ShapeProfiler] 直接启动 vllm serve (无模板脚本)", flush=True)
                print(f"[ShapeProfiler] 按 Ctrl+C 停止所有 DP worker 并生成报告", flush=True)

                def _run_dp_worker_direct(dp_rank, vllm_engine_port, visible_devices,
                                          model, dp_size, dp_address, dp_rpc_port,
                                          tp_size, output_dir, host, gpu_mem_util,
                                          max_num_seqs, max_model_len):
                    """在子进程中直接启动 vllm serve"""
                    os.environ["SHAPE_PROFILER_OUTPUT_DIR"] = output_dir
                    os.environ["SHAPE_PROFILER_DP_RANK"] = str(dp_rank)
                    os.environ["SHAPE_PROFILER_DP_SIZE"] = str(dp_size)
                    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices

                    serve_cmd = [
                        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                        "--model", model,
                        "--host", host,
                        "--port", str(vllm_engine_port),
                        "--trust-remote-code",
                        "--enforce-eager",
                        "--gpu-memory-utilization", str(gpu_mem_util),
                        "--max-num-seqs", str(max_num_seqs),
                        "--tensor-parallel-size", str(tp_size),
                        "--data-parallel-size", str(dp_size),
                        "--data-parallel-rank", str(dp_rank),
                        "--data-parallel-address", dp_address,
                        "--data-parallel-rpc-port", dp_rpc_port,
                    ]
                    if max_model_len:
                        serve_cmd.extend(["--max-model-len", str(max_model_len)])

                    print(f"[ShapeProfiler] DP rank {dp_rank}: "
                          f"vllm serve 启动 (port={vllm_engine_port}, "
                          f"devices={visible_devices})", flush=True)
                    subprocess.run(serve_cmd, check=True)

                processes = []
                for i in range(dp_size_local):
                    dp_rank = dp_rank_start + i
                    vllm_engine_port = vllm_start_port + i
                    visible_devices = ",".join(
                        str(x) for x in range(i * tp_size, (i + 1) * tp_size))

                    p = multiprocessing.Process(
                        target=_run_dp_worker_direct,
                        args=(dp_rank, vllm_engine_port, visible_devices,
                              args.model, dp_size, dp_address, dp_rpc_port,
                              tp_size, output_dir_abs, args.host,
                              args.gpu_memory_utilization, args.max_num_seqs,
                              args.max_model_len),
                    )
                    processes.append((dp_rank, p))
                    p.start()
                    print(f"[ShapeProfiler] DP rank {dp_rank}: 进程已启动 "
                          f"(pid={p.pid}, devices={visible_devices}, port={vllm_engine_port})",
                          flush=True)

                try:
                    for dp_rank, p in processes:
                        p.join()
                except KeyboardInterrupt:
                    print(f"\n[ShapeProfiler] 收到中断信号，正在停止所有 DP worker...",
                          flush=True)
                    for dp_rank, p in processes:
                        if p.is_alive():
                            p.terminate()
                    for dp_rank, p in processes:
                        p.join(timeout=15)
                        if p.is_alive():
                            p.kill()
                            p.join()
                        print(f"[ShapeProfiler] DP rank {dp_rank}: 进程已停止", flush=True)
                finally:
                    import time
                    time.sleep(2)
                    _profiler.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    _profiler.generate_reports(output_dir_abs, merge_pids=True)
                    if "SHAPE_PROFILER_OUTPUT_DIR" in os.environ:
                        del os.environ["SHAPE_PROFILER_OUTPUT_DIR"]

        # ==================== 普通 Serve 模式 (单进程) ====================
        elif args.serve:
            serve_cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", args.model,
                "--host", args.host,
                "--port", str(args.port),
                "--trust-remote-code",
                "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--max-num-seqs", str(args.max_num_seqs),
                "--tensor-parallel-size", str(args.tensor_parallel_size),
            ]
            if args.max_model_len:
                serve_cmd.extend(["--max-model-len", str(args.max_model_len)])
                
                                # "--enforce-eager",

            print(f"[ShapeProfiler] 启动 vLLM serve 模式", flush=True)
            print(f"[ShapeProfiler] 模型: {args.model}", flush=True)
            print(f"[ShapeProfiler] 地址: {args.host}:{args.port}", flush=True)
            print(f"[ShapeProfiler] tensor_parallel_size: {args.tensor_parallel_size}", flush=True)
            print(f"[ShapeProfiler] Hooks 将在 worker 进程中安装", flush=True)
            print(f"[ShapeProfiler] 按 Ctrl+C 停止服务并生成报告", flush=True)

            process = None
            def _sigterm_handler(signum, frame):
                raise KeyboardInterrupt
            import signal as _signal
            _signal.signal(_signal.SIGTERM, _sigterm_handler)
            try:
                process = subprocess.Popen(serve_cmd)
                process.wait()
            except KeyboardInterrupt:
                print(f"\n[ShapeProfiler] 收到中断信号，正在停止服务...", flush=True)
                if process:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            except Exception as e:
                print(f"[ShapeProfiler] 服务出错: {e}", flush=True)
                import traceback
                traceback.print_exc()
            finally:
                # 等待 worker 子进程写完 .jsonl 文件
                import time as _time
                _time.sleep(3)
                _profiler.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _profiler.generate_reports(args.output_dir, merge_pids=True)
                if "SHAPE_PROFILER_OUTPUT_DIR" in os.environ:
                    del os.environ["SHAPE_PROFILER_OUTPUT_DIR"]

        # ==================== 离线推理模式 ====================
        else:
            try:
                from vllm import LLM, SamplingParams

                llm_kwargs = {
                    "model": args.model,
                    "trust_remote_code": True,
                    "enforce_eager": False,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "max_num_seqs": args.max_num_seqs,
                    "tensor_parallel_size": args.tensor_parallel_size,
                }
                if args.max_model_len:
                    llm_kwargs["max_model_len"] = args.max_model_len

                print(f"[ShapeProfiler] 加载模型: {args.model}", flush=True)
                print(f"[ShapeProfiler] 使用 tensor_parallel_size={args.tensor_parallel_size}", flush=True)

                is_single_card = args.tensor_parallel_size == 1
                if is_single_card:
                    print(f"[ShapeProfiler] 单卡模式: 在主进程安装 hooks，实时写入文件", flush=True)
                    _profiler.install_hooks(args.output_dir, is_single_card=True)
                else:
                    print(f"[ShapeProfiler] 多卡模式: Hooks 将通过 patch_shape_profiler 在 worker 进程中安装", flush=True)

                llm = LLM(**llm_kwargs)

                sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
                outputs = llm.generate([args.prompt], sampling_params)

                for output in outputs:
                    print(f"生成结果: {output.outputs[0].text[:100]}...")

            except ImportError as e:
                print(f"vLLM 未安装: {e}", flush=True)
            except Exception as e:
                print(f"推理出错: {e}", flush=True)
                import traceback
                traceback.print_exc()
            finally:
                _profiler.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if args.tensor_parallel_size == 1:
                    _profiler.uninstall_hooks()
                else:
                    print(f"[ShapeProfiler] 等待 worker 进程完成文件写入...", flush=True)
                    import time
                    import gc
                    if 'llm' in locals():
                        del llm
                    gc.collect()
                    time.sleep(2)
                _profiler.generate_reports(args.output_dir, merge_pids=(args.tensor_parallel_size > 1))
                if "SHAPE_PROFILER_OUTPUT_DIR" in os.environ:
                    del os.environ["SHAPE_PROFILER_OUTPUT_DIR"]
    else:
        print("请指定 --model 参数 (或使用 --merge-only 模式)")
        print("")
        print("示例:")
        print("  # 离线推理模式 (单卡)")
        print("  python shape_profiler.py --model Qwen/Qwen2-7B --prompt 'Hello'")
        print("")
        print("  # Serve 模式 (单进程)")
        print("  python shape_profiler.py --model Qwen/Qwen2-7B --serve --port 8000")
        print("  # 按 Ctrl+C 停止服务后自动生成报告")
        print("")
        print("  # DP Serve 模式 (使用模板脚本)")
        print("  python shape_profiler.py --model /path/to/model --serve \\")
        print("    --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 \\")
        print("    --dp-address 141.61.52.167 --dp-rpc-port 12321 --vllm-start-port 8000 \\")
        print("    --dp-template ./run_dp_template.sh --output-dir ./output")
        print("")
        print("  # DP Serve 模式 (直接启动 vllm serve)")
        print("  python shape_profiler.py --model /path/to/model --serve \\")
        print("    --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 \\")
        print("    --dp-address 141.61.52.167 --dp-rpc-port 12321 --vllm-start-port 8000 \\")
        print("    --output-dir ./output")
        print("")
        print("  # 手动 DP 模式 (配合 launch_online_dp.py 使用)")
        print("  export SHAPE_PROFILER_OUTPUT_DIR=./output")
        print("  python launch_online_dp.py --dp-size 32 --tp-size 1 ...")
        print("  # 运行完毕后合并报告:")
        print("  python shape_profiler.py --merge-only --output-dir ./output")
