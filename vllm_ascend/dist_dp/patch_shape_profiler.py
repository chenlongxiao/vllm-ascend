"""
Shape Profiler Patch for vLLM Ascend

使用方式:
    # 单卡/TP 模式:
    export SHAPE_PROFILER_OUTPUT_DIR="./output"
    python your_inference_script.py

    # 分布式 DP 模式 (配合 launch_online_dp.py):
    export SHAPE_PROFILER_OUTPUT_DIR="./output"
    python launch_online_dp.py --dp-size 32 --tp-size 1 ...
    # 每个 DP rank 的 worker 进程会自动检测 dp_rank 并生成独立的记录文件
    # 文件命名: shape_records_dp{dp_rank}_pid{pid}.jsonl

    # 或通过 shape_profiler.py 启动 DP 模式:
    python shape_profiler.py --model /path/to/model --serve --dp-size 32 ...
"""

import atexit
import functools
import json
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _detect_dp_info():
    """
    从环境变量或命令行参数中检测 DP rank 和 DP size。

    检测优先级:
    1. SHAPE_PROFILER_DP_RANK / SHAPE_PROFILER_DP_SIZE 环境变量 (由 shape_profiler.py 设置)
    2. sys.argv 中的 --data-parallel-rank / --data-parallel-size 参数 (vllm serve 模式)
    3. VLLM_DP_RANK / VLLM_DP_SIZE 环境变量 (离线 DP 模式)
    """
    dp_rank = -1
    dp_size = 1

    # 优先级 1: shape_profiler 专用环境变量
    env_rank = os.environ.get("SHAPE_PROFILER_DP_RANK")
    if env_rank is not None:
        try:
            dp_rank = int(env_rank)
        except ValueError:
            pass

    env_size = os.environ.get("SHAPE_PROFILER_DP_SIZE")
    if env_size is not None:
        try:
            dp_size = int(env_size)
        except ValueError:
            pass

    # 优先级 2: 从 sys.argv 解析 (vllm serve --data-parallel-rank X)
    if dp_rank == -1:
        for i, arg in enumerate(sys.argv):
            if arg == "--data-parallel-rank" and i + 1 < len(sys.argv):
                try:
                    dp_rank = int(sys.argv[i + 1])
                except ValueError:
                    pass
            elif arg.startswith("--data-parallel-rank="):
                try:
                    dp_rank = int(arg.split("=", 1)[1])
                except ValueError:
                    pass

    if dp_size <= 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--data-parallel-size" and i + 1 < len(sys.argv):
                try:
                    dp_size = int(sys.argv[i + 1])
                except ValueError:
                    pass
            elif arg.startswith("--data-parallel-size="):
                try:
                    dp_size = int(arg.split("=", 1)[1])
                except ValueError:
                    pass

    # 优先级 3: vLLM 原生 DP 环境变量
    if dp_rank == -1:
        vllm_rank = os.environ.get("VLLM_DP_RANK")
        if vllm_rank is not None:
            try:
                dp_rank = int(vllm_rank)
            except ValueError:
                pass

    if dp_size <= 1:
        vllm_size = os.environ.get("VLLM_DP_SIZE")
        if vllm_size is not None:
            try:
                dp_size = int(vllm_size)
            except ValueError:
                pass

    return dp_rank, dp_size


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
        if not torch.compiler.is_compiling():
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
        if not torch.compiler.is_compiling():
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
        if not torch.compiler.is_compiling():
            grid_info = str(self._grid_key) if self._grid_key else "unknown"
            self._profiler.record_call(f"{self._op_name}[{grid_info}]", args, kwargs, result)
        return result
    
    def __getattr__(self, name):
        return getattr(self._original, name)
    
    def __repr__(self):
        return f"_TritonLaunchedKernelWrapper({self._original})"


class _ShapeDispatchMode(TorchDispatchMode):
    """在图模式 graph replay 阶段捕获所有 dispatch op 的 shape/dtype。"""

    def __init__(self, profiler: "ShapeProfiler"):
        super().__init__()
        self._profiler = profiler

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if not torch.compiler.is_compiling() and self._profiler.enabled:
            op_name = str(func)
            arg_shapes = []
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor):
                    arg_shapes.append({"arg_idx": i, "shape": [int(d) for d in a.shape], "dtype": str(a.dtype)})
            out_shapes = []
            for item in (result if isinstance(result, (list, tuple)) else [result]):
                if isinstance(item, torch.Tensor):
                    out_shapes.append({"shape": [int(d) for d in item.shape], "dtype": str(item.dtype)})
            if arg_shapes:
                self._profiler._save_record_directly(op_name, args, kwargs or {}, result)
                self._profiler.record_call(op_name, args, kwargs or {}, result)
        return result


class ShapeProfiler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.enabled = False
        self.original_funcs: Dict[str, Tuple[Any, Any]] = {}
        self.records: List[Dict] = []
        self._seen_keys: set = set()
        self.output_dir: str = None
        self._pid = os.getpid()
        self._dp_rank, self._dp_size = _detect_dp_info()
        self._is_dp_mode = self._dp_rank >= 0 and self._dp_size > 1
        self._dispatch_mode: "_ShapeDispatchMode | None" = None
        atexit.register(self._cleanup)
    
    def _get_shape_dtype(self, obj: Any) -> List[Tuple[List[int], str]]:
        results = []
        if isinstance(obj, torch.Tensor):
            results.append(([int(d) for d in obj.shape], str(obj.dtype)))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                results.extend(self._get_shape_dtype(item))
        elif isinstance(obj, dict):
            for v in obj.values():
                results.extend(self._get_shape_dtype(v))
        return results
    
    def _make_dedup_key(self, record: Dict) -> tuple:
        def arg_key(a):
            return (a.get("arg_idx", a.get("key")),
                    tuple((tuple(t["shape"]), t["dtype"]) for t in a.get("tensors", [])))
        return (
            record["op_name"],
            tuple(arg_key(a) for a in record.get("arg_shapes", [])),
            tuple(arg_key(a) for a in record.get("kwarg_shapes", [])),
            tuple((tuple(o["shape"]), o["dtype"]) for o in record.get("output_shapes", [])),
        )

    def _store_record(self, record: Dict):
        key = self._make_dedup_key(record)
        if key not in self._seen_keys:
            self._seen_keys.add(key)
            self.records.append(record)
    
    def _get_record_filename(self):
        if self._is_dp_mode:
            return f"shape_records_dp{self._dp_rank}_pid{self._pid}.jsonl"
        return f"shape_records_pid{self._pid}.jsonl"

    def save_records_to_file(self):
        if not self.records or not self.output_dir:
            return
        output_file = os.path.join(self.output_dir, self._get_record_filename())
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

        arg_shapes = []
        for i, arg in enumerate(args):
            shapes_dtypes = self._get_shape_dtype(arg)
            if shapes_dtypes:
                arg_shapes.append({
                    "arg_idx": i,
                    "tensors": [{"shape": s, "dtype": d} for s, d in shapes_dtypes],
                })

        kwarg_shapes = []
        for k, v in kwargs.items():
            shapes_dtypes = self._get_shape_dtype(v)
            if shapes_dtypes:
                kwarg_shapes.append({
                    "key": k,
                    "tensors": [{"shape": s, "dtype": d} for s, d in shapes_dtypes],
                })

        output_shapes = [{"shape": s, "dtype": d} for s, d in self._get_shape_dtype(result)]

        record = {
            "op_name": op_name,
            "arg_shapes": arg_shapes,
            "kwarg_shapes": kwarg_shapes,
            "output_shapes": output_shapes,
            "pid": self._pid,
        }

        if self._is_dp_mode:
            record["dp_rank"] = self._dp_rank
            record["dp_size"] = self._dp_size

        self._store_record(record)
    
    def _wrap_function(self, func: Callable, op_name: str) -> Callable:
        profiler = self
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            if not torch.compiler.is_compiling():
                profiler.record_call(op_name, args, kwargs, result)
            return result
        wrapped._shape_profiler_wrapped = True
        return wrapped
    
    def _save_record_directly(self, op_name: str, args: tuple, kwargs: dict, result: Any):
        """直接保存 shape 信息到文件，不依赖 enabled 状态"""
        output_dir = os.environ.get("SHAPE_PROFILER_OUTPUT_DIR")
        if not output_dir:
            return

        arg_shapes = []
        for i, arg in enumerate(args):
            shapes_dtypes = self._get_shape_dtype(arg)
            if shapes_dtypes:
                arg_shapes.append({
                    "arg_idx": i,
                    "tensors": [{"shape": s, "dtype": d} for s, d in shapes_dtypes],
                })

        kwarg_shapes = []
        for k, v in kwargs.items():
            shapes_dtypes = self._get_shape_dtype(v)
            if shapes_dtypes:
                kwarg_shapes.append({
                    "key": k,
                    "tensors": [{"shape": s, "dtype": d} for s, d in shapes_dtypes],
                })

        output_shapes = [{"shape": s, "dtype": d} for s, d in self._get_shape_dtype(result)]

        record = {
            "op_name": op_name,
            "arg_shapes": arg_shapes,
            "kwarg_shapes": kwarg_shapes,
            "output_shapes": output_shapes,
            "pid": self._pid,
        }

        if self._is_dp_mode:
            record["dp_rank"] = self._dp_rank
            record["dp_size"] = self._dp_size

        key = self._make_dedup_key(record)
        if key in self._seen_keys:
            return
        self._seen_keys.add(key)

        output_file = os.path.join(output_dir, self._get_record_filename())
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                # print(f"[ShapeProfiler] 直接保存文件: {output_file}", flush=True)
                f.flush()
        except Exception as e:
            print(f"[ShapeProfiler] 直接保存记录{output_file}失败: {e}", flush=True)
    
    def _wrap_triton_function(self, func: Callable, op_name: str) -> Callable:
        profiler = self
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            if not torch.compiler.is_compiling():
                profiler._save_record_directly(op_name, args, kwargs, result)
                profiler.record_call(op_name, args, kwargs, result)
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
            
            wrapped = self._wrap_triton_function(original_func, op_name)

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
            _call_count = [0]
            
            def wrapped_register(op_name, op_func, mutates_args=None, fake_impl=None, target_lib=None, dispatch_key=None, tags=()):
                _call_count[0] += 1
                print(f"[ShapeProfiler] DEBUG: direct_register_custom_op 被调用 (第{_call_count[0]}次): {op_name}, pid={os.getpid()}", flush=True)
                wrapped_func = profiler._wrap_triton_function(op_func, f"custom_op.{op_name}")
                print(f"[ShapeProfiler] DEBUG: 成功包装 custom op: {op_name}", flush=True)
                result = original_register(op_name, wrapped_func, mutates_args, fake_impl, target_lib, dispatch_key, tags)
                print(f"[ShapeProfiler] DEBUG: 完成注册 custom op: {op_name}", flush=True)
                return result
            
            import vllm.utils.torch_utils
            vllm.utils.torch_utils.direct_register_custom_op = wrapped_register
            self.original_funcs["direct_register_custom_op"] = (vllm.utils.torch_utils, original_register)
            print(f"[ShapeProfiler] DEBUG: 成功 hook direct_register_custom_op, pid={os.getpid()}", flush=True)
        except ImportError as e:
            print(f"[ShapeProfiler] DEBUG: hook direct_register_custom_op 失败 (ImportError): {e}", flush=True)
        except Exception as e:
            print(f"[ShapeProfiler] DEBUG: hook direct_register_custom_op 失败: {e}", flush=True)
    
    def _is_graph_mode(self):
        return "--enforce-eager" not in sys.argv

    def install_hooks(self, output_dir: str = None):
        if self.enabled:
            return

        self.output_dir = output_dir
        self.enabled = True

        graph_mode = self._is_graph_mode()
        if graph_mode:
            print("[ShapeProfiler] DEBUG: 检测到图模式，跳过 torch ops hook 以避免 graph break", flush=True)

        self._hook_direct_register_custom_op()
        
        triton_ops = [
            ("vllm_ascend.ops.triton.rope", "rope_forward_triton", "rope_forward_triton"),
            ("vllm_ascend.ops.triton.rope", "rope_forward_triton_siso", "rope_forward_triton_siso"),
            ("vllm_ascend.ops.triton.reject_sample", "rejection_greedy_sample_with_triton", "rejection_greedy_sample_with_triton"),
            ("vllm.v1.sample.rejection_sampler", "sample_recovered_tokens_kernel", "sample_recovered_tokens_kernel"),
            ("vllm.v1.sample.rejection_sampler", "expand_kernel", "expand_kernel"),
            ("vllm.v1.sample.rejection_sampler", "rejection_sample", "rejection_sample"),
            ("vllm.model_executor.layers.mamba.ops.causal_conv1d", "causal_conv1d_fn", "causal_conv1d_fn"),
            ("vllm_ascend.ops.triton.mamba.causal_conv1d", "causal_conv1d_update_npu", "causal_conv1d_update_npu"),
            ("vllm_ascend.ops.triton.fla.l2norm", "l2norm_fwd", "l2norm_fwd"),
            # ("vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope", "split_qkv_rmsnorm_rope_impl", "split_qkv_rmsnorm_rope_impl"),
            ("vllm_ascend.ops.triton.fla.chunk", "chunk_gated_delta_rule", "chunk_gated_delta_rule"),
            ("vllm.model_executor.layers.fla.ops.chunk", "chunk_scaled_dot_kkt_fwd", "chunk_scaled_dot_kkt_fwd"),
            ("vllm_ascend.ops.triton.fla.chunk", "chunk_scaled_dot_kkt_fwd", "chunk_scaled_dot_kkt_fwd"),
            ("vllm_ascend.ops.triton.fla.cumsum", "chunk_local_cumsum_scalar", "chunk_local_cumsum_scalar"),
            ("vllm.model_executor.layers.fla.ops", "fused_recurrent_gated_delta_rule", "fused_recurrent_gated_delta_rule"),
            ("vllm_ascend.ops.triton.muls_add", "muls_add_triton", "muls_add"),
            ("vllm.model_executor.layers.layernorm", "fused_add_rms_norm", "fused_add_rms_norm"),
            ("vllm_ascend.ops.triton.fla.solve_tril", "solve_tril", "solve_tril"),
            ("vllm_ascend.ops.triton.fla.chunk", "solve_tril", "solve_tril"),
            ("vllm_ascend.ops.triton.fla.wy_fast", "recompute_w_u_fwd", "recompute_w_u_fwd"),
            ("vllm_ascend.ops.triton.fla.chunk", "recompute_w_u_fwd", "recompute_w_u_fwd"),
            ("vllm_ascend.ops.triton.fla.chunk_o", "chunk_fwd_o", "chunk_fwd_o"),
            ("vllm_ascend.ops.triton.layernorm_gated", "layer_norm_fwd_npu", "layer_norm_fwd_npu"),
            ("vllm_ascend.ops.triton.fused_gdn_gating", "fused_gdn_gating_patch", "fused_gdn_gating_patch"),
            ("vllm.model_executor.layers.mamba.gdn_linear_attn", "fused_gdn_gating", "fused_gdn_gating"),
        ]
        
        for module_path, attr_name, op_name in triton_ops:
            self._install_module_hook(module_path, attr_name, op_name)

        if not graph_mode:
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
            ("torch.ops._C_ascend.dispatch_ffn_combine", "dispatch_ffn_combine"),
            ("torch.ops.vllm.all_reduce", "all_reduce"),
            ("torch.ops.vllm.all_gather", "all_gather"),
            # ("torch.ops.vllm.maybe_all_gather_and_maybe_unpad", "maybe_all_gather_and_maybe_unpad"),
            ("torch.ops.mie_ops.npu_mla_preprocess", "npu_mla_preprocess"),
            ("torch_npu.npu_rms_norm", "npu_rms_norm"),
            ("torch_npu.npu_rotary_mul", "npu_rotary_mul"),
            ("torch_npu.npu_dynamic_mx_quant", "npu_dynamic_mx_quant"),
            ("torch_npu.npu_apply_rotary_pos_emb", "npu_apply_rotary_pos_emb"),
            ("torch_npu.npu_quantize", "npu_quantize"),
            ("torch_npu.npu_dynamic_quant", "npu_dynamic_quant"),
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

            self._patch_cross_module_refs()

        else:
            self._dispatch_mode = _ShapeDispatchMode(self)
            self._dispatch_mode.__enter__()
            print("[ShapeProfiler] DEBUG: 图模式已启用 TorchDispatchMode 捕获 torch ops", flush=True)

        print(f"[ShapeProfiler] Worker 进程已安装 {len(self.original_funcs)} 个 hooks"
              f" (pid={self._pid}, dp_rank={self._dp_rank}, dp_size={self._dp_size})", flush=True)

    def _patch_cross_module_refs(self):
        for op_name, (parent_module, original_func) in list(self.original_funcs.items()):
            if not callable(original_func):
                continue
            wrapped_func = getattr(parent_module, original_func.__name__, None)
            if wrapped_func is None or not getattr(wrapped_func, "_shape_profiler_wrapped", False):
                continue
            for mod_name, mod in list(sys.modules.items()):
                if mod is None or mod is parent_module:
                    continue
                try:
                    mod_dict = vars(mod)
                except TypeError:
                    continue
                for attr_key, attr_val in list(mod_dict.items()):
                    if attr_val is original_func:
                        setattr(mod, attr_key, wrapped_func)
                        print(f"[ShapeProfiler] DEBUG: 修补跨模块引用: {mod_name}.{attr_key} -> {op_name}", flush=True)
    
    def uninstall_hooks(self):
        if self._dispatch_mode is not None:
            try:
                self._dispatch_mode.__exit__(None, None, None)
            except Exception:
                pass
            self._dispatch_mode = None
        for op_name, (parent, original_func) in self.original_funcs.items():
            if hasattr(original_func, "__name__"):
                setattr(parent, original_func.__name__, original_func)
        self.original_funcs.clear()
        self.enabled = False
    
    def _cleanup(self):
        if self.enabled:
            self.uninstall_hooks()
        if self.records and self.output_dir:
            self.save_records_to_file()


_profiler = ShapeProfiler()


def install_worker_hooks():
    dp_rank, dp_size = _detect_dp_info()
    print(f"[ShapeProfiler] DEBUG: install_worker_hooks 被调用, pid={os.getpid()}, "
          f"dp_rank={dp_rank}, dp_size={dp_size}", flush=True)

    output_dir = os.environ.get("SHAPE_PROFILER_OUTPUT_DIR")
    if output_dir:
        print(f"[ShapeProfiler] DEBUG: 输出目录: {output_dir}", flush=True)
    else:
        print(f"[ShapeProfiler] DEBUG: 未设置输出目录，记录将不会保存到文件", flush=True)

    _profiler.install_hooks(output_dir=output_dir)


print(f"[ShapeProfiler] DEBUG: patch_shape_profiler 模块被导入, pid={os.getpid()}, "
      f"dp_info={_detect_dp_info()}", flush=True)
install_worker_hooks()
