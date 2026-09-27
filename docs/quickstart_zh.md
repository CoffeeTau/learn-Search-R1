# 0. 先建立一句话认识

Search-R1 不是一个“接上搜索结果再回答”的普通 RAG 项目，而是一个用强化学习训练搜索 Agent 的项目：模型需要自己决定什么时候输出搜索动作、用什么查询词，以及什么时候停止搜索并给出最终答案。

一次最典型的轨迹是：

```text
问题
  -> 模型思考
  -> <search>检索词</search>
  -> 检索服务返回 <information>文档</information>
  -> 模型结合新信息继续思考
  -> <answer>最终答案</answer>
  -> 规则奖励判断答案是否命中
  -> PPO / GRPO 更新模型
```

第一次上手不要直接从八卡训练开始。建议按下面三个成功标准逐级推进：

1. **5 分钟：**看懂一条轨迹怎样获得奖励。
2. **30 分钟：**用仓库自带的小语料启动 BM25 检索 API，并真正请求一次 `/retrieve`。
3. **GPU 服务器：**用已训练模型跑推理；确认闭环后再启动 PPO 或 GRPO。

# 1. 先认清五个核心部件

| 部件 | 作用 | 主要入口 |
|---|---|---|
| QA 数据 | 提供问题、标准答案及数据来源 | `scripts/data_process/nq_search.py` |
| 策略模型 | 生成搜索动作或最终答案 | `search_r1/llm_agent/generation.py` |
| 检索服务 | 接收查询并返回候选文档 | `search_r1/search/retrieval_server.py` |
| 规则奖励 | 从轨迹中取最终答案并做 Exact Match | `verl/utils/reward_score/qa_em.py` |
| RL 训练器 | 采样轨迹、计算优势、更新策略 | `verl/trainer/main_ppo.py`、`verl/trainer/ppo/ray_trainer.py` |

这里最容易混淆的是“模型”和“搜索引擎”。模型不会直接读取 Wikipedia 索引；它只输出文本形式的 `<search>` 动作。`LLMGenerationManager` 解析动作后，通过 HTTP 请求独立运行的检索服务，再把结果作为新观察交还给模型。

# 2. 五分钟零依赖实验：看见奖励是怎样产生的

这个实验只调用仓库里的奖励源码，不需要 Torch、模型或 GPU。请在仓库根目录执行：

```bash
python3 - <<'PY'
import runpy

qa_em = runpy.run_path("verl/utils/reward_score/qa_em.py")

# 第一个 <answer> 来自数据提示词中的格式示例；奖励代码取最后一个答案。
trajectory = """
For example, <answer> Beijing </answer>.
<think>I have enough evidence.</think>
<answer>Pavia Cathedral</answer>
"""

answer = qa_em["extract_solution"](trajectory)
score = qa_em["em_check"](answer, ["Pavia Cathedral"])

print("extracted answer:", answer)
print("exact-match reward:", score)
PY
```

预期输出：

```text
extracted answer: Pavia Cathedral
exact-match reward: 1
```

再把最后一个答案改成 `Pavia`，奖励会变成 `0`。这说明当前主训练入口使用的是规范化后的**完整字符串精确匹配**：会忽略大小写、英文冠词、标点和多余空格，但不会把“部分命中”当成正确答案。

还有一个不直观的实现细节：`extract_solution()` 要求整条解码文本至少出现两个 `<answer>` 块，原因是训练提示词本身已经带有 `<answer> Beijing </answer>` 这个格式示例，它再取最后一个作为模型答案。若以后重写提示词并删掉这个示例，奖励提取逻辑也要一起调整。

**思考**

如果模型找到了正确证据，却输出 `<answer>Pavia</answer>`，这条轨迹会得到正奖励吗？

<details>
<summary><strong>参考分析</strong></summary>

不会。当前奖励只检查最终答案文本是否与标准答案 Exact Match，并不单独奖励“搜到了好文档”或“推理过程合理”。这也是结果奖励稀疏的含义。

</details>

# 3. 三十分钟最小闭环：启动本地检索 API

这一阶段不下载完整 Wikipedia，也不启动大模型。目标只是用 `example/corpus.jsonl` 的十条示例文档完成：

```text
本地 JSONL -> 建 BM25 索引 -> FastAPI 服务 -> HTTP 查询 -> 返回文档
```

## 3.1 环境要求

推荐 Linux 或 macOS、Conda、Python 3.10，并确保下面两条命令成功：

```bash
conda --version
java -version
```

Pyserini 依赖 Java。如果 `java -version` 失败，应先安装 Pyserini 当前版本所要求的 JDK。建议把检索环境和训练环境分开，避免 Torch、FAISS、vLLM 的版本互相牵制。

```bash
conda create -n searchr1-retriever python=3.10 -y
conda activate searchr1-retriever

pip install torch transformers datasets pyserini faiss-cpu uvicorn fastapi
```

这里选择 BM25 是因为它不需要 GPU；仓库中的 e5 查询编码器会直接调用 CUDA，不适合作为 CPU 入门路径。

## 3.2 用小语料建索引

仍在仓库根目录执行：

```bash
mkdir -p .quickstart/indexes

python search_r1/search/index_builder.py \
  --retrieval_method bm25 \
  --corpus_path example/corpus.jsonl \
  --save_dir .quickstart/indexes
```

成功标志是看到 `Start building bm25 index...` 和 `Finish!`，并生成 `.quickstart/indexes/bm25/`。

## 3.3 启动服务

在终端 A 执行：

```bash
conda activate searchr1-retriever

python search_r1/search/retrieval_server.py \
  --index_path .quickstart/indexes/bm25 \
  --corpus_path example/corpus.jsonl \
  --topk 3 \
  --retriever_name bm25
```

看到 Uvicorn 监听 `http://0.0.0.0:8000` 后，不要关闭终端 A。

## 3.4 真正请求一次检索接口

在终端 B 执行：

```bash
curl -s http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Pavia Cathedral dome"],"topk":3,"return_scores":true}' \
  | python -m json.tool
```

成功标志不是 HTTP 200 本身，而是响应满足以下结构，并且候选文档中能看到 `Pavia Cathedral`：

```json
{
  "result": [
    [
      {
        "document": {
          "contents": "..."
        },
        "score": 0.0
      }
    ]
  ]
}
```

分数值取决于索引和查询，不应照抄上面的占位值。完成后可在终端 A 按 `Ctrl+C` 停止服务。

到这里，你已经独立跑通了 Search-R1 Agent 的“工具侧”。训练和推理使用的也是同一个 `/retrieve` JSON 协议。

# 4. 跑已训练模型：看见完整 Reasoning + Search

这一步建议使用带 NVIDIA GPU 的 Linux 服务器。当前仓库给出的兼容组合是 Python 3.9、Torch 2.4.0 + CUDA 12.1、vLLM 0.6.3；不要直接使用系统里更新得多的 Python 版本碰运气。

## 4.1 建立模型环境

```bash
conda create -n searchr1 python=3.9 -y
conda activate searchr1

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb
```

先做环境预检：

```bash
python -V
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import verl, vllm, transformers; print('imports ok')"
```

## 4.2 准备完整检索库

在空间充足的位置保存索引和语料：

```bash
export SEARCHR1_CORPUS_DIR=/path/to/searchr1-corpus
mkdir -p "$SEARCHR1_CORPUS_DIR"

python scripts/download.py --save_path "$SEARCHR1_CORPUS_DIR"
cat "$SEARCHR1_CORPUS_DIR"/part_* > "$SEARCHR1_CORPUS_DIR/e5_Flat.index"
gzip -d "$SEARCHR1_CORPUS_DIR/wiki-18.jsonl.gz"
```

下载后应至少存在：

```bash
ls -lh \
  "$SEARCHR1_CORPUS_DIR/e5_Flat.index" \
  "$SEARCHR1_CORPUS_DIR/wiki-18.jsonl"
```

修改 `retrieval_launch.sh` 中的 `file_path`，令它指向 `SEARCHR1_CORPUS_DIR` 的实际值。然后在检索环境的终端 A 启动：

```bash
conda activate searchr1-retriever
bash retrieval_launch.sh
```

注意：这个脚本使用 e5、FAISS GPU 和 `--faiss_gpu`，因此检索进程也需要可见 GPU。若只是调试协议，继续使用第 3 节的 BM25 即可。

## 4.3 运行推理

先打开 `infer.py` 看两个最重要的配置：

```python
question = "..."
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
```

确认检索服务仍在 8000 端口后，在终端 B 执行：

```bash
conda activate searchr1
python infer.py
```

观察重点不是答案本身，而是输出是否出现以下交替结构：

```text
<think>...</think>
<search>...</search>
<information>...</information>
<think>...</think>
<answer>...</answer>
```

`infer.py` 没有显式的最大搜索轮数保护。如果模型持续搜索而不结束，可先按 `Ctrl+C`，再检查模型、问题和检索结果是否匹配。

# 5. 最后再跑训练

## 5.1 数据如何变成训练样本

以 NQ 为例：

```bash
conda activate searchr1
python scripts/data_process/nq_search.py --local_dir ./data/nq_search
```

每条 Parquet 样本的关键字段是：

```text
prompt                         模型看到的问题与动作格式说明
reward_model.ground_truth      标准答案列表
data_source                    决定使用哪一个奖励函数
extra_info.index               样本编号；GRPO 用它识别同题多条轨迹
```

可先抽查一条，避免等到训练启动后才发现数据错误：

```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset("parquet", data_files="data/nq_search/train.parquet", split="train")
print(ds[0])
PY
```

## 5.2 PPO 路径

根目录的 `train_ppo.sh` 当前配置为单机 8 卡、Llama-3.2-3B、NQ 数据和两轮搜索：

```bash
conda activate searchr1
wandb login
bash train_ppo.sh
```

启动前至少核对这些值：

```text
CUDA_VISIBLE_DEVICES
DATA_DIR
BASE_MODEL
trainer.n_gpus_per_node
data.train_batch_size
actor_rollout_ref.actor.ppo_micro_batch_size
retriever.url
```

不要只把 `trainer.n_gpus_per_node=8` 改成较小数字就认为单卡一定能跑；批大小、微批大小、模型并行和显存占用也要一起缩放。先以“能完成一个训练 step”为目标，再逐步恢复吞吐配置。

## 5.3 GRPO 路径

仓库根目录的 `train_grpo.sh` 定义了 `DATA_DIR`，但命令里读取的是未定义的 `TRAIN_DATA_DIR` 和 `TEST_DATA_DIR`，不建议直接运行。更完整的现成入口是：

```bash
bash scripts/nq_hotpotqa/v0.1/train_grpo.sh
```

该脚本预期 `data/nq_hotpotqa_train/train.parquet` 和 `test.parquet` 已存在，并将同一个问题复制为多条采样轨迹（`n_agent=5`），再按同题轨迹之间的相对结果计算 GRPO 优势。第一次实验可复制脚本到自己的实验目录再改参数，保留原始脚本作为对照。

**思考**

为什么 GRPO 需要同一道题生成多条轨迹，而只生成一条时很难得到“组内相对优势”？

<details>
<summary><strong>参考分析</strong></summary>

当前实现按样本 `index` 对轨迹分组，再用同组奖励的均值和标准差做标准化。一组只有一条轨迹时，代码只能把它当作均值为 0、标准差为 1 的特殊情况，无法比较“这条搜索策略比同题其他策略好多少”。

</details>

# 6. 读代码时沿这一条链路走

不要一开始通读整个 veRL。按下面顺序阅读，能最快把数据、Agent 和 RL 串起来：

1. `scripts/data_process/nq_search.py`：问题怎样包装成 prompt，标准答案放在哪里。
2. `search_r1/llm_agent/generation.py` 的 `run_llm_loop()`：多轮生成循环。
3. 同文件的 `postprocess_predictions()`：正则怎样识别 `<search>` 与 `<answer>`。
4. 同文件的 `batch_search()`：查询怎样 POST 到 `/retrieve`。
5. `search_r1/search/retrieval_server.py` 的 `retrieve_endpoint()`：API 怎样返回文档。
6. `verl/trainer/main_ppo.py` 的 `RewardManager`：轨迹怎样变成终点奖励。
7. `verl/utils/reward_score/qa_em.py`：最终答案怎样抽取和匹配。
8. `verl/trainer/ppo/ray_trainer.py`：奖励怎样进入 advantage 和 actor update。

一个重要边界是：`<information>` 是环境提供的状态，不是模型自己生成的动作。启用 `state_masking=true` 时，训练器通过 `info_mask` 把这些检索结果 token 从策略损失中遮掉，避免让模型为外部搜索结果本身背梯度责任。

# 7. 常见卡点与定位顺序

| 现象 | 优先检查 | 原因 |
|---|---|---|
| `Connection refused` | `curl http://127.0.0.1:8000/retrieve` 是否可达 | 检索服务没启动或端口不一致 |
| BM25 启动时报 Java 错误 | `java -version` | Pyserini/Lucene 依赖 JDK |
| e5 检索报 CUDA 错误 | GPU 是否可见、是否误走 CPU | 当前编码器直接调用 `.cuda()` |
| 推理只搜索、不回答 | 检索结果质量、问题是否匹配、模型是否正确 | `infer.py` 没有最大轮数上限 |
| 奖励一直为 0 | 最终标签、答案文本、标准答案字段 | 当前是规则型 Exact Match |
| GRPO 没有有效组内差异 | 同题轨迹数和 `extra_info.index` | 相对优势依赖同题多样采样 |
| 一启动就 OOM | micro batch、序列长度、并行配置 | 官方脚本面向 8 卡实验配置 |
| 根目录 GRPO 找不到数据 | `TRAIN_DATA_DIR`、`TEST_DATA_DIR` | 根脚本变量名与声明不一致 |

定位时遵守从便宜到昂贵的顺序：

```text
文件存在
  -> Python / CUDA / Java 版本
  -> import 成功
  -> 检索 API 单测
  -> 单条推理
  -> 单个训练 step
  -> 长时间训练
```

# 8. 一页总结

- Search-R1 的关键不是“有搜索”，而是让模型把搜索当成可学习的文本动作。
- Agent 循环只识别 `<search>` 和 `<answer>`；搜索结果通过 `<information>` 回填。
- 检索服务与模型训练是两个进程，通过 `/retrieve` 解耦。
- 当前 QA 主线使用结果级 Exact Match 奖励，正确证据和漂亮推理不会自动得到额外奖励。
- PPO 使用 critic 估值；GRPO 依赖同题多轨迹的相对奖励。
- 最稳妥的上手顺序是：奖励 smoke test → 小语料 BM25 API → 已训练模型推理 → 单步训练 → 完整实验。

# 9. 本次验证边界

截至 2026-09-27，本仓库在当前机器上的只读环境检查结果是：Apple Silicon、Python 3.14、没有 `nvidia-smi`，且 Torch、Transformers、Datasets、Ray、FAISS、FastAPI、Pyserini 均未安装。因此本次已经完成源码链路核对、零依赖奖励实验和静态语法检查；没有把 GPU 推理、检索服务或 RL 训练表述为已在当前机器实跑。完整闭环需要按第 3～5 节在相应环境继续验证。
