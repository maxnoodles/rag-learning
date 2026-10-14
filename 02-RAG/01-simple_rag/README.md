# 可观察的本地 RAG 学习项目

使用 25 组真实 AI/RAG 历史聊天，观察 TXT/Markdown 解析、LangChain 递归切分、本地 Google embedding、原问题与 Codex 抽象问题两轮 FAISS + BM25 混合检索、候选合并、Cross-Encoder 重排、上下文合并和 Codex 回答。当前命令行入口每次重建索引；存储层仍支持按指纹复用磁盘缓存。

## 直接运行

首次使用先下载一次重排模型（默认保存在 Git 忽略的 `artifacts/models/bge-reranker-base`）：

```sh
../.venv/bin/python scripts/download_reranker.py
```

随后在这个目录运行：

```sh
./run.sh
./run.sh --query '强化学习的奖励必须由人类每次给出吗？'
./run.sh --rebuild
./run.sh --retrieve-only
```

`--rebuild` 指定强制重建；当前入口已默认每次重建。`--retrieve-only` 仅跳过最终回答，仍使用已登录的 Codex CLI 账号抽象问题；无需另外提供 API key。账号或服务异常会明确报错，不会伪造答案。

默认测试问题来自已有聊天：FAISS IndexFlatIP 中 Flat / IP 的含义，以及归一化与余弦相似度的关系。

## 终端会展示什么

1. 原始问题、缓存是否命中及重建原因。
2. Embedding 设备与维度，重建时的文档和 chunk 数量。
3. 第一轮召回数量、Codex 抽象后的问题、第二轮召回数量、Merge 去重后的数量，以及 Cross-Encoder 最终选中片段的来源、命中轮次和 vector / BM25 / RRF / CrossEncoder 分数。
4. 上下文去重、补全或预算不足的处理说明，以及实际送给模型的证据正文和 `[E1]` 等来源。
5. Codex 回答、状态和引用。

存储层缓存命中时跳过解析、切分和文档 embedding，仅按需计算两轮 query embedding；BM25 从同一快照的原文和元数据在内存重建。终端不重复打印原文、每个 chunk、全部候选正文和 prompt 指令，不再计算或输出阶段与总耗时。

使用 `--output-json artifacts/my-run.json` 可保存两轮各自的查询与候选、Codex 抽象调用信息、Merge 结果、全部重排分数、最终选择、证据、完整 prompt 和回答；详细报告序列化和额外的上下文 token 计数仅在指定此参数时执行，报告不含耗时字段。

## 递归多级切块

使用 LangChain `RecursiveCharacterTextSplitter`，依次尝试段落、换行、中英文句子、子句标点、空格，最后退化到字符。每组问答独立切分，embedding 输入重复保留该组问题与标题。

`--chunk-tokens` 限制的是完整 embedding 输入（含前缀和特殊 token），以模型 tokenizer 计数；`--overlap-tokens` 是正文重叠的目标上限。先预留前缀和重叠预算，再递归切分不重叠的核心块，最后从原文加入前一块的 token 后缀并检查完整预算。这样能准确保存字符 start/end，重复文本和空白也不会造成定位歧义；前缀过长时可能缩短重叠。

参考 [LangChain 递归分割文档](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)。

## BM25 与向量混合检索

同一批 chunks 同时进入 FAISS 和 BM25 倒排索引。BM25 检索正文、标题和本组问题，采用 jieba 搜索分词、英文大小写归一化，使用 k1=1.5、b=0.75、正值 IDF `log(1 + (N-df+0.5)/(df+0.5))`。没有匹配词的 chunk 不进入 BM25 候选。

两路各召回至多 `fetch-k` 个候选，按 chunk ID 去重，并使用加权 RRF（Reciprocal Rank Fusion）排序：

```text
RRF = vector_weight / (rrf_k + vector_rank)
    + (1 - vector_weight) / (rrf_k + bm25_rank)
```

未进入某一路候选的项在该路贡献为 0。默认向量/BM25 权重各 0.5，rrf_k=60。每轮融合后保留至多 `top-k` 个候选；不直接相加量纲不同的 BM25 分数与余弦分数。JSON 保留两路排名、原始分数和 RRF 分数。

混合检索的 relevance 为 RRF 按理论首位上限归一化后的值：`RRF × (rrf_k+1) / 有候选的通道权重之和`，范围为 [0,1]。`--vector-weight 1` 为纯向量召回；`--vector-weight 0` 为纯 BM25 召回，两者仍执行两轮检索和 Cross-Encoder 重排。

## 两轮检索与 Cross-Encoder 重排

```text
原始问题 → FAISS + BM25 → RRF → top_k=4
    ↓ Codex 抽象问题（只读原始问题，保留实体和必要约束，不生成答案）
抽象问题 → FAISS + BM25 → RRF → top_k=4
    ↓
Merge：按 chunk ID 合并去重，至多 8 个候选，保留命中轮次
    ↓
Cross-Encoder（原始问题 + 每个候选）→ top_n=2
    ↓
上下文合并与预算 → Codex 回答原始问题
```

两轮固定使用同一索引快照。即使抽象问题与原问题相同，也会执行第二轮；重复候选仅重排一次。Merge 不额外截断、不直接相加两轮的原始分数。合并项的检索分数来自首次命中的轮次，各轮完整分数分别记录在 JSON 的 `retrieval_rounds` 中。

Google EmbeddingGemma 2 继续负责向量召回；本地 `BAAI/bge-reranker-base` 对 query/document 联合编码，输出单个相关性 logit。对全部合并候选使用**原始问题**打分，再按分数降序取 `top-n`，同分按合并顺序保持稳定；分数不是正确概率。重排不再经过 MMR，旧 `retrieve()` 与评估脚本只保留为单轮 MMR 基线。

模型使用 float32；`--device` 同时作用于 embedding 和重排，默认优先 MPS。重排 batch 默认 4，每对文本最多 512 个重排 tokenizer tokens，过长时由模型 tokenizer 截断；它与 embedding 的 tokenizer 预算不同，长候选尾部可能未参与打分。模型只从本地加载，查询时不会自动联网下载权重。模型用法见 [BGE 官方模型卡](https://huggingface.co/BAAI/bge-reranker-base) 与 [CrossEncoder 文档](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)。

选中的 chunk 再按同一问答组聚合；能放入预算时补全整组问答，放不下则保留命中范围。`top-n` 限制的是选中 chunk 数，补全后的证据组数可能更少。抽象、模型加载或打分失败会明确报错，不自动退回旧流程；指定 JSON 报告时保留已完成的轮次结果。

## 可调整参数

```sh
./run.sh --fetch-k 20 --top-k 4 --top-n 2
./run.sh --vector-weight 0.5 --rrf-k 60
./run.sh --retrieve-only --vector-weight 1  # 纯向量对照
./run.sh --chunk-tokens 512 --overlap-tokens 64 --context-tokens 6000
./run.sh --docs /absolute/path/to/documents
./run.sh --model-path /absolute/path/to/embeddinggemma-2
./run.sh --reranker-model-path /absolute/path/to/bge-reranker-base --reranker-batch-size 4
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

聊天正文、索引、SDK 检查文件和测试日志均在 Git 忽略目录。没有上传到 GitHub。问题抽象阶段只把原始问题交给 Codex 服务；回答阶段提交原始问题及筛选后的上下文。

## 缓存机制

`data/indexes/generations/<id>/` 存放 `vectors.faiss`、`metadata.sqlite`、`manifest.json`；`CURRENT` 指向已发布版本。

缓存指纹覆盖文档内容、模型权重及配置、embedding/分词/切分依赖版本、parser/chunker/BM25 版本和切分参数。缓存命中还要校验磁盘文件哈希、向量维度和 ID/原文范围一致性。正文修改、删除、新增，或模型与切分配置变更都会重建。问题抽象、Cross-Encoder、RRF 与两路权重参数只影响查询，不改变存储层指纹；当前命令行入口每次重建。

构建失败保留上一成功 generation；不会用不匹配的旧快照偷偷回答新文档。单机文件锁避免两个进程同时发布，原子切换 CURRENT 保持请求内一致性。本学习版保留旧 generation，暂不自动清理，频繁重建会占用额外磁盘。

## 环境与测试

本章共用 `02-RAG/.venv`（Python 3.12）；`01-simple_rag` 及后续小节都使用它，小节不再单独建立 `.venv`。Embedding 和 Cross-Encoder 共用已有的 `sentence-transformers`，模型均使用本地目录。

新机器在 `02-RAG` 目录运行：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

另行提供本地模型目录和已登录的 Codex CLI。默认模型路径针对当前机器。

Cursor 推荐直接打开 `02-RAG`，选择 `.venv/bin/python`；按 F5 可运行或调试小节。仓库根目录不放 Python 配置，请打开章节目录。`.vscode` 和虚拟环境均由 Git 忽略。切换到其他章节时，使用该章节自己的 `.venv`。

SDK 抽象与回答阶段均禁用 shell、MCP、web 等工具并检查返回活动；来源引用做格式和编号校验，语义是否真的被证据支持仍需评估。

```sh
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python scripts/evaluate_retrieval.py  # 旧版单轮 RRF + MMR 基线
```

详见 [完成报告](reports/COMPLETION_REPORT.md) 和 [测试报告](reports/TEST_REPORT.md)。该评估脚本保留旧版单轮 RRF + MMR 基线，不能用来宣称新流程的质量；6 个手选问题的功能检查不代表正式检索 benchmark。新流程可用 `--output-json` 检查两轮候选与全部重排分数。
