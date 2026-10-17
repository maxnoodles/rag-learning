# 基础 RAG 全流程

对应本目录的基础 RAG 封板实现。下图使用默认参数，候选数量均为上限；两轮检索按顺序执行，并共用同一索引快照。

```mermaid
flowchart TD
    START(["运行 run.sh / run.py"])

    subgraph INDEX["一、数据处理与存储 · prepare_index()"]
        SOURCE["保留原始 TXT / Markdown 文件"]
        DOC["解析问答组、标题与来源"]
        CHUNK["递归切成 chunks<br/>完整 embedding 输入上限 512 tokens<br/>正文重叠目标上限 64 tokens"]
        EMB["本地 EmbeddingGemma 2<br/>生成归一化文档向量"]
        FAISS[("vectors.faiss<br/>FAISS IndexFlatIP 向量索引")]
        META[("metadata.sqlite<br/>解析后的原文、chunks、来源与字符范围")]
        BM25["同一批 chunks 经 jieba 分词<br/>在内存构建 BM25 索引"]
        SNAP["统一检索快照<br/>向量、片段、来源与 BM25"]

        SOURCE --> DOC --> CHUNK
        CHUNK --> EMB --> FAISS
        DOC --> META
        CHUNK --> META
        META --> BM25
        FAISS --> SNAP
        META --> SNAP
        BM25 --> SNAP
    end

    START -->|"当前命令行入口每次强制重建"| SOURCE

    subgraph SEARCH["二、问题解析与检索 · process_query()"]
        Q["用户原始问题"]
        R1["第一轮：原问题检索<br/>问题 embedding → FAISS，最多 10 条<br/>问题文本分词 → BM25，最多 10 条"]
        F1["加权 RRF 融合<br/>向量 / BM25 权重各 0.5，rrf_k = 60<br/>保留最多 4 条"]
        ABSTRACT["Codex 抽象原问题<br/>保留核心意图、实体、否定关系与必要约束"]
        R2["第二轮：抽象问题检索<br/>问题 embedding → 同一 FAISS，最多 10 条<br/>问题文本分词 → 同一 BM25，最多 10 条"]
        F2["加权 RRF 融合<br/>参数同第一轮，保留最多 4 条"]
        MERGE["按 chunk ID 合并去重 → 最多 8 条<br/>保留最强轮次的召回分数与全部命中轮次"]
        MMR["MMR 压缩 → 最多 4 条<br/>兼顾召回相关性与候选间多样性<br/>lambda = 0.7"]
        RERANK["本地 BGE Cross-Encoder 重排<br/>原始问题 + 每个候选联合编码打分<br/>按相关性分数保留最多 2 个 chunks"]

        Q --> R1 --> F1
        F1 -->|"第一轮完成后"| ABSTRACT
        ABSTRACT --> R2 --> F2
        F1 --> MERGE
        F2 --> MERGE
        MERGE --> MMR --> RERANK
    end

    SNAP --> R1
    SNAP -. 同一索引 .-> R2

    subgraph ANSWER["三、证据整理与回答 · build_context() / generate_answer()"]
        CONTEXT["按选中 chunks 回查原文<br/>优先补全同组完整问答<br/>超预算则合并重叠或相邻命中范围"]
        BUDGET["证据预算 6000 tokens<br/>放不下的证据省略<br/>添加 E1、E2 等引用编号与来源"]
        ONLY{"retrieve-only？"}
        STOP(["输出检索结果与证据"])
        ENOUGH{"预算内有可用证据？"}
        INSUFFICIENT(["返回 insufficient：无可用证据"])
        PROMPT["构建 Prompt<br/>原始问题 + 整理后的证据 + 回答约束"]
        GEN["Codex 仅依据证据回答<br/>事实结论附引用，证据不足或冲突时明确说明"]
        CHECK["校验结构化输出与引用编号<br/>正文引用与引用列表必须一致"]
        OUT(["输出答案、状态与来源<br/>answered / insufficient / conflict"])

        CONTEXT --> BUDGET --> ONLY
        ONLY -->|"是"| STOP
        ONLY -->|"否"| ENOUGH
        ENOUGH -->|"否"| INSUFFICIENT
        ENOUGH -->|"是"| PROMPT --> GEN --> CHECK --> OUT
    end

    RERANK --> CONTEXT
    SNAP -. 原文与来源 .-> CONTEXT
```

## 数据处理与存储

先统一切块，再为同一批 chunks 建立两路检索能力。Embedding 模型负责把文本编码成向量；BM25 是关键词检索算法，两者都不负责切块。

磁盘快照包含 `vectors.faiss`、`metadata.sqlite` 与 `manifest.json`，通过 `CURRENT` 指向当前版本。FAISS 文件保存向量索引，SQLite 保存解析后的文档、chunks 与映射关系。BM25 不单独保存索引文件，而是在创建或加载快照时，由 SQLite 对应的同一批 chunks、标题与本组问题在内存中构建。

当前命令行入口强制重建索引；底层存储仍支持指纹校验和缓存复用。

## 问题解析与检索

第一轮使用原始问题，第二轮使用 Codex 抽象后的问题。每轮都先执行向量与 BM25 两路召回，再用 RRF 按排名融合；两轮结果合并时按 chunk ID 去重，不直接累加两轮的原始分数。

MMR 使用最强命中轮次的召回相关性，以及候选 embedding 之间的余弦相似度，压缩重复内容。随后，`BAAI/bge-reranker-base` Cross-Encoder 对“原始问题 + 候选文本”联合打分。Cross-Encoder 分数不是余弦相似度，也不是正确概率。

## 证据整理与回答生成

最终最多选中 2 个 chunks，随后可能补全整组问答，因此实际提供给大语言模型的证据可能比这两个片段更长，证据组数也可能少于 2。6000 tokens 是证据内容的估算预算，不是完整模型请求的预算。

`--retrieve-only` 仍会调用 Codex 抽象问题，只跳过最终回答。模型加载、抽象、生成或引用校验失败时，流程明确报错；指定 `--output-json` 时会保存已完成的过程与错误信息。引用编号校验不等于验证每个结论的语义正确性。

## 代码入口

| 阶段 | 对应代码 |
| --- | --- |
| 主流程、参数与两轮调用顺序 | [run.py](run.py) |
| 文档解析与递归切块 | [document_parser.py](rag/document_parser.py)、[chunker.py](rag/chunker.py) |
| 文档与问题向量编码 | [embedder.py](rag/embedder.py) |
| FAISS / SQLite 快照与 BM25 构建 | [vector_store.py](rag/vector_store.py)、[lexical_index.py](rag/lexical_index.py) |
| RRF、两轮合并与 MMR | [retriever.py](rag/retriever.py) |
| Cross-Encoder 重排 | [reranker.py](rag/reranker.py) |
| 证据整理与 Prompt | [context_builder.py](rag/context_builder.py) |
| Codex 问题抽象、回答与引用校验 | [codex_adapter.py](rag/codex_adapter.py) |
