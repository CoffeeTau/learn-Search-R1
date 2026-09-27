modelscope download \
  --dataset hhjinjiajie/FlashRAG_Dataset \
  --local_dir ./data/FlashRAG_Dataset \
  --include 'nq/*.jsonl'

下载后检查：

ls -lh data/FlashRAG_Dataset/nq/

应至少看到：

train.jsonl
test.jsonl

我已经修改了项目里的 scripts/data_process/nq_search.py，增加了本地 ModelScope 数据支持。把更新后的文件同步到服务器，然后执行：

python scripts/data_process/nq_search.py \
  --source_dir ./data/FlashRAG_Dataset \
  --local_dir ./data/nq_search

检查最终数据：

ls -lh data/nq_search/train.parquet data/nq_search/test.parquet

成功生成后直接运行：

bash train_grpo.sh

这里要区分两个目录：

data/FlashRAG_Dataset/   # ModelScope 下载的原始 JSONL
data/nq_search/          # 项目转换后的训练 Parquet，即 DATA_DIR