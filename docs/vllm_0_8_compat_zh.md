# 服务器使用方式

这条兼容路径面向已经安装 **Torch 2.6.0 + vLLM 0.8.x**、不能重新下载大包的服务器。它保留原来的 FSDP 训练和 Search-R1 多轮 Agent，只替换 rollout 与权重同步实现。

先把修改后的仓库复制到服务器。在已有环境中不要让 pip 解析或下载依赖：

```bash
cd /path/to/learn-Search-R1
pip install -e . --no-deps
```

执行只读预检：

```bash
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=0
python scripts/check_vllm_compat.py
```

预期至少看到：

```text
torch        2.6.0
vllm         0.8.4
cuda_available True
OK: package imports and the vLLM compatibility surface look usable
```

启动训练前继续保留：

```bash
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=0
```

然后运行原 PPO/GRPO 脚本。第一次建议把训练步数和数据量降到最小，只验证一个 rollout 和一个参数更新；不要直接开始长任务。

# 为什么必须关闭 V1

vLLM 0.8.4 同时提供 V0 和 V1 引擎。当前回移植使用官方 veRL 0.8 迁移方案中的 external launcher、sleep/wake 和运行时 `load_weights()`，但本仓库的旧 FSDP 混合引擎仍通过 V0 的 model executor 访问模型。因此这里明确使用 `VLLM_USE_V1=0`，而不是假装 V1 已被完整适配。

# 首次失败时需要保留的日志

如果预检通过而训练仍失败，请保存从启动开始到第一个 traceback 结束的完整日志，并额外执行：

```bash
python -m pip check
python -c "import torch, vllm, transformers, tensordict, ray; print(torch.__version__, vllm.__version__, transformers.__version__, tensordict.__version__, ray.__version__)"
nvidia-smi
```

这三组信息可以区分代码 API、CUDA/驱动以及已有依赖冲突。由于当前开发机没有 NVIDIA GPU，本次只能完成源码级和模拟接口验证；服务器上的首个 rollout 才是运行兼容性结论。
