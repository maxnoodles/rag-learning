# 可观察的本地 RAG 学习项目

使用 25 组真实 AI/RAG 历史聊天，观察 TXT/Markdown 解析、问答切分、本地 Google embedding、FAISS 检索、MMR、上下文合并和 Codex 回答。首次运行建库，后续自动复用磁盘缓存。

## 直接运行

在这个目录运行：

```sh
./run.sh
./run.sh --query '强化学习的奖励必须由人类每次给出吗？'
./run.sh --rebuild
./run.sh --retrieve-only
```

`--rebuild` 强制重建。`--retrieve-only` 跳过 Codex，只观察本地检索；其他情况下不需要 API key，使用已登录的 Codex CLI 账号。账号或服务异常会明确报错，不会伪造答案。

默认测试问题来自已有聊天：FAISS IndexFlatIP 中 Flat / IP 的含义，以及归一化与余弦相似度的关系。

## 终端会展示什么

1. 原始问题、缓存是否命中及重建原因。
2. Embedding 设备与维度，重建时的文档和 chunk 数量。
3. 召回与 MMR 选择数量，选中片段的来源及 relevance / redundancy / MMR 分数。
4. 上下文去重、补全或预算不足的处理说明，以及实际送给模型的证据正文和 `[E1]` 等来源。
5. Codex 回答、状态和引用。

缓存命中时跳过解析、切分和文档 embedding，仅计算 query embedding。终端不重复打印原文、每个 chunk、全部候选正文和 prompt 指令，不再计算或输出阶段与总耗时。

使用 `--output-json artifacts/my-run.json` 可保存详细候选、MMR 选择、证据、完整 prompt 和回答；详细报告序列化和额外的上下文 token 计数仅在指定此参数时执行，报告不含耗时字段。

## MMR 是什么

MMR = **Maximal Marginal Relevance（最大边际相关性）**。

候选先按 query 相似度召回，第一项选最相关者；后续选择综合考虑与 query 的相关性、与已选 chunk 的最大相似度：

```text
MMR = λ × relevance − (1 − λ) × redundancy
```

默认 λ=0.7。λ=1 相当于只按相关性排序；λ 较小更重视多样性。MMR 不等于精确文本去重，不保证每个多样化结果都与问题有关。选择后的 chunk 再按同一问答组聚合；能放入预算时补全整组问答，放不下则保留命中范围，绝不跨文档拼接。

## 可调整参数

```sh
./run.sh --fetch-k 20 --top-k 4 --mmr-lambda 0.7
./run.sh --chunk-tokens 512 --overlap-tokens 64 --context-tokens 6000
./run.sh --docs /absolute/path/to/documents
./run.sh --model-path /absolute/path/to/embeddinggemma-2
./run.sh --codex-model MODEL_ID --timeout 300
./run.sh --output-json artifacts/my-run.json
```

切分使用 embedding tokenizer，包含标题与问题前缀。上下文预算使用独立的 `o200k_base` 估算，因为 Codex 默认模型没有固定；这不是整个 SDK 请求的精确 token 数。实际 SDK 请求还包含自己的系统提示和协议开销，运行报告保存服务返回的 usage。预算是证据块预算，不是完整模型窗口上限。

## 样例与来源

`data/private/chat_samples/001.md` 至 `025.md` 是真实用户问题及助手最终回答，不是重新生成的知识点。保留前一条用户问题作为必要上下文，以及聊天标题、Asia/Shanghai 时间、thread ID 和原始 JSONL 行号。

覆盖 5 个聊天：轻量 embedding 模型、合成数据与过拟合、企业交互训练、强化学习、向量数据库。排除了系统/开发者消息、工具输出、内部推理、过程 commentary 和机器记忆引用尾注。历史助手回答仅代表当时聊天内容，不视为已经验证的当前事实。

可重新导出同一批样例：

```sh
../.venv/bin/python scripts/export_chat_samples.py
```

聊天正文、索引、SDK 检查文件和测试日志均在 Git 忽略目录。没有上传到 GitHub。正常回答阶段仅把筛选后的上下文交给 Codex 服务。

## 缓存机制

`data/indexes/generations/<id>/` 存放 `vectors.faiss`、`metadata.sqlite`、`manifest.json`；`CURRENT` 指向已发布版本。

缓存指纹覆盖文档内容、模型权重及配置、embedding 依赖版本、parser/chunker 版本和切分参数。缓存命中还要校验磁盘文件哈希、向量维度和 ID/原文范围一致性。正文修改、删除、新增，或模型与切分配置变更都会重建。MMR 参数只影响查询，不触发文档重新编码。

构建失败保留上一成功 generation；不会用不匹配的旧快照偷偷回答新文档。单机文件锁避免两个进程同时发布，原子切换 CURRENT 保持请求内一致性。本学习版保留旧 generation，暂不自动清理，频繁重建会占用额外磁盘。

## 环境与测试

本章共用 `02-RAG/.venv`（Python 3.12）；`01-simple_rag` 及后续小节都使用它，小节不再单独建立 `.venv`。当前依赖已复制到章节环境，已移除指向历史 gemma 环境的 `.pth`，不再依赖那个环境。未重新下载模型或 Python 包。

新机器在 `02-RAG` 目录运行：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

另行提供本地模型目录和已登录的 Codex CLI。默认模型路径针对当前机器。

Cursor 推荐直接打开 `02-RAG`，选择 `.venv/bin/python`；按 F5 可运行或调试小节。仓库根目录不放 Python 配置，请打开章节目录。`.vscode` 和虚拟环境均由 Git 忽略。切换到其他章节时，使用该章节自己的 `.venv`。

SDK 生成阶段禁用 shell、MCP、web 等工具并检查返回活动；来源引用做格式和编号校验，语义是否真的被证据支持仍需评估。

```sh
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python scripts/evaluate_retrieval.py
```

详见 [完成报告](reports/COMPLETION_REPORT.md) 和 [测试报告](reports/TEST_REPORT.md)。检索评估为 6 个手选问题的功能检查，不代表正式检索 benchmark。
