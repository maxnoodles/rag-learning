# 基础 RAG

已于 2026-10-15 [封板](FREEZE.md)，保留为学习与对照基线。后续高级能力在 [02-high_level_rag](../02-high_level_rag/README.md) 开发。

## 运行

本章共用 `02-RAG/.venv`，安装方式见 [章节说明](../README.md)。在本目录运行：

```sh
./run.sh
./run.sh --query '强化学习的奖励必须由人类每次给出吗？'
./run.sh --retrieve-only
./run.sh --output-json artifacts/my-run.json
./run.sh --help
```

需要本地 EmbeddingGemma 2、BGE 重排模型和已登录的 Codex CLI。默认 embedding 路径针对当前机器，可用 `--model-path` 覆盖。重排模型默认位于 `artifacts/models/bge-reranker-base`；新环境缺少模型时运行：

```sh
../.venv/bin/python scripts/download_reranker.py
```

模型在查询时只从本地加载。`--retrieve-only` 跳过最终回答，仍会调用 Codex 抽象问题；抽象阶段提交原问题，回答阶段提交原问题与筛选后的证据。服务失败会明确报错。

## 流程

完整流程图与三阶段说明见 [基础 RAG 全流程](FLOW.md)，覆盖数据存储、两轮混合检索、MMR、Cross-Encoder、证据整理及带引用回答。

TXT / Markdown 按问答组递归切分 → 本地 EmbeddingGemma 2 编码 → FAISS 向量索引与 SQLite 元数据；同一批片段建立 BM25 索引。

```text
原问题   → FAISS + BM25 → RRF → top_k=4
Codex 抽象问题 → 同一索引检索 → RRF → top_k=4
    ↓
按 chunk ID 合并去重（最多 8），保留最强轮次分数与全部命中轮次
    ↓
MMR 压缩 → mmr_k=4
    ↓
BGE Cross-Encoder 按原问题重排 → top_n=2
    ↓
同组问答补全 / 命中范围合并 → 上下文预算 → Codex 带引用回答
```

RRF 按两路排名加权融合，默认向量与 BM25 各占 0.5；MMR 使用召回相关性与片段间 embedding 余弦相似度去冗余，公式为 `lambda × 相关性 - (1-lambda) × 重复度`。Cross-Encoder 仅对 MMR 保留的候选联合编码并打分。

终端打印两轮查询、候选数量、最终片段分数、实际证据和答案。`--output-json` 额外保存各轮候选、合并结果、MMR 选择、重排分数、完整 prompt 与答案。

## 常用参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--fetch-k` | 10 | 每轮向量 / BM25 各自召回上限 |
| `--top-k` | 4 | 每轮 RRF 保留上限 |
| `--mmr-k` | 4 | 合并后交给重排模型的上限 |
| `--top-n` | 2 | 重排后最终片段上限 |
| `--mmr-lambda` | 0.7 | MMR 相关性权重 |
| `--vector-weight` | 0.5 | RRF 向量权重；0 为纯 BM25，1 为纯向量 |
| `--rrf-k` | 60 | RRF 排名平滑常数 |
| `--chunk-tokens` / `--overlap-tokens` | 512 / 64 | 完整 embedding 输入上限 / 正文重叠目标上限 |
| `--context-tokens` | 6000 | 证据预算，使用 `o200k_base` 估算 |
| `--device` | auto | 优先 MPS，也可指定 cpu |

数量约束：`1 <= top-k <= fetch-k`，`1 <= top-n <= mmr-k <= 2 * top-k`。更多路径和模型参数见 `./run.sh --help`。

## 目录与数据

- `run.py` / `run.sh`：主流程与启动入口。
- `rag/`：解析、切块、模型适配、存储、检索和上下文处理。
- `scripts/download_reranker.py`：首次下载重排模型。
- `tests/`：流程和边界回归测试。
- `data/private/chat_samples/`：默认使用的 25 份真实历史问答及来源清单，保留原问题、回答和来源信息；历史回答不代表已验证事实。
- `data/indexes/`：自动生成的索引，可删除后重建。
- `artifacts/models/`、`artifacts/tiktoken-cache/`：本地重排模型和分词缓存；其余运行报告按需生成。

私有语料、模型、索引、缓存及运行报告均由 Git 忽略。可用 `--docs /absolute/path/to/documents` 换成自己的 TXT / Markdown。

当前入口每次强制重建索引；存储层仍支持指纹校验、缓存复用和原子发布。历史索引版本不会自动回收，可在停止运行后清理 `data/indexes/`。

## 验证与限制

```sh
../.venv/bin/python -m unittest discover -s tests -v
```

MMR 可能提前丢弃必要证据，重排输入超过 512 个重排 tokenizer tokens 时会截断。上下文预算不含完整 SDK 协议开销；引用编号校验也不能证明答案语义正确。默认参数与少量样例不构成质量基准。
