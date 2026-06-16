# 单机多卡
将patch_shape_profiler.py放到vllm-ascend/ops目录下，
并修改vllm_ascend/ops/__init__.py,将以下两行写入__init__文件开头
```shell
import vllm.utils.torch_utils
import vllm_ascend.ops.patch_shape_profiler
```
设置环境变量用于决定数据落盘地址
```shell
export SHAPE_PROFILER_OUTPUT_DIR=/path/to/output   # 必须，触发 profiler 
```
正常执行单机服务拉起命令，根据业务实际场景修改:
```shell
export VLLM_DISABLE_COMPILE_CACHE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
export ASCEND_RT_VISIBLE_DEVICES=4,5
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_IF_IP="141.61.73.211"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024

export OMP_NUM_THREADS=1
#echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
#sysctl -w vm.swappiness=0
#sysctl -w kernel.numa_balancing=0
#sysctl kernel.sched_migration_cost_ns=50000
#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
#export VLLM_VERSION=0.17.0
#export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /home/weights/Qwen3.5-35B-A3B-W8A8-MXFP8-FULL-QUANT \
    --served-model-name "qwen3.5" \
    --host 0.0.0.0 \
    --port 7777 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --max-model-len 133120 \
    --max-num-batched-tokens 16384  \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
    --trust-remote-code \
    --async-scheduling \
    --allowed-local-media-path / \
    --quantization ascend \
    --mm-processor-cache-gb 0 \
    --additional-config '{"enable_cpu_binding":true}' \
    --no-enable-prefix-caching
```
  正常发送推理请求后，Ctrl+C结束服务，然后进行合并报告：
```shell
python /vllm-workspace/shape_profiler.py --merge-only --output-dir /path/to/output
```
  
# 多机
将patch_shape_profiler.py放到要采集的节点的vllm-ascend/ops目录下，
并修改vllm_ascend/ops/__init__.py,将以下两行写入__init__文件开头
```shell
import vllm.utils.torch_utils
import vllm_ascend.ops.patch_shape_profiler
```
设置环境变量
```shell
export SHAPE_PROFILER_OUTPUT_DIR=/path/to/output   # 必须，触发 profiler 
```
正常启动分布式DP命令（该命令根据业务场景变更）
```shell
python launch_online_dp.py     --dp-size 2     --tp-size 1     --dp-address 127.0.0.1     --dp-rpc-port 12325     --vllm-start-port 8300
```
以及proxy
```shell
python /vllm-workspace/dp_load_balance_proxy_server.py \
    --host 0.0.0.0 --port 9000 \
    --dp-hosts 127.0.0.1 127.0.0.1 \
    --dp-ports 8300 8301
```
正常发送推理请求后，Ctrl+C结束服务，然后进行合并报告：
```shell
python /vllm-workspace/shape_profiler.py --merge-only --output-dir /path/to/output
```
