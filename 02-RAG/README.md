# RAG 学习项目

本章按版本保留可运行的学习基线，高级能力在新目录中迭代。

| 目录 | 定位 | 状态 |
| --- | --- | --- |
| [01-simple_rag](01-simple_rag/README.md) | 基础 RAG：文档切分、两轮混合检索、MMR、Cross-Encoder、带引用回答 | 2026-10-15 功能封板，保留为对照基线 |
| [02-high_level_rag](02-high_level_rag/README.md) | Graph RAG：递归切块、Codex 抽取与实体选择、BFS 子图检索 | 已实现图文档构建与检索 |

## 版本边界

`01-simple_rag` 的封板范围与已知限制见 [封板记录](01-simple_rag/FREEZE.md)。后续新功能在 `02-high_level_rag` 实现；基础版仅处理必要的缺陷修复与文档勘误，并记录影响。高级版的实验结果以基础版作为对照。

## 共享环境

本章各小节共用 `02-RAG/.venv`，依赖入口为本目录的 `requirements.txt`。在本目录初始化：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

基础版和 Graph RAG 的运行方式见各自 README；新增依赖时需验证基础版仍能在共享环境中运行。各版本独立保存索引与运行产物，模型权重、私有语料和本地产物继续由 Git 忽略。
