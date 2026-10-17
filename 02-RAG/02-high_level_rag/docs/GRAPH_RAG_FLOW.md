# Graph RAG 流程图

更新日期：2026-10-17。本文保存本项目当前流程，以及讨论中的生产环境扩展设计。Mermaid 代码块可在 GitHub 中直接查看为流程图。

## 当前实现：图文档构建与 BFS 检索

当前用 JSON 持久化 GraphDocument，检索时构建内存邻接表。节点以 `(type, id)` 区分；候选号仅在当前候选列表中使用。尚未接入图数据库、实体向量索引、人工别名表或最终答案生成。

```mermaid
flowchart TD
    D["原始文档"] --> C["LangChain 递归切块：按 token 计数"]
    C --> X["LLMGraphTransformer 调用 Codex 抽取实体、关系和引文"]
    X --> GD["GraphDocument：nodes、relationships、source"]
    GD --> V["检查节点、关系方向、端点和原文引文"]
    V --> J["保存 graph_documents.json"]
    J --> KG["加载内存图：合并节点和三元组，保留全部来源"]
    KG --> E["全部已有实体：候选号、类型、名称"]
    Q["用户问题 question"] --> S["Codex 从候选列表选择起点实体"]
    E --> S
    S --> CHECK["校验候选号存在性和起点数量"]
    CHECK --> FOUND{"是否有匹配实体？"}
    FOUND -->|无| EMPTY["返回 no_match 和空子图"]
    FOUND -->|有| BFS["多源 BFS：默认双向、最多 2 跳、30 个节点"]
    KG --> BFS
    BFS --> SUB["保留已发现节点集合内的全部原始有向边"]
    SUB --> OUT["返回起点、分层节点、关系、引文和源 chunk"]
    OUT --> REPORT["保存检索报告 JSON 并展示结果"]
```

- 建图入口：[`run.py`](../run.py)，主要流程：[`pipeline.py`](../graph_rag/pipeline.py)。
- 检索入口：[`retrieve.py`](../retrieve.py)，实体选择和 BFS：[`retriever.py`](../graph_rag/retriever.py)。
- BFS 的入边、出边或双向设置控制节点发现范围，输出的关系始终保留原方向。节点达到上限时报告截断。
- 子图是候选证据集合，可能含有无关事实；当前没有根据问题筛选关系或生成答案。

## 生产环境扩展：建图与存储设计

以下是设计方案，尚未实现。三元组是“主体—关系—客体”，例如“张三—负责—知识助手项目”；GraphDocument 是中间结构，生成该对象不等于已经写入图数据库。

```mermaid
flowchart TD
    RAW["文档与业务数据"] --> CHUNK["解析、切块并保存来源信息"]
    CHUNK --> EXTRACT["抽取实体、关系、属性和原文证据"]
    EXTRACT --> STRUCT["GraphDocument 等中间结构"]
    STRUCT --> VALIDATE["按 schema 校验类型、关系和证据"]
    VALIDATE --> RESOLVE["实体解析：识别同名、别名，合并同一实体"]
    RESOLVE --> PROFILE["维护稳定 entity_id、标准名称、别名和实体简介"]
    PROFILE --> GRAPH["图存储：实体、关系、属性、来源引用"]
    PROFILE --> NAME["名称与别名索引、全文索引"]
    PROFILE --> VECTOR["实体简介的 embedding 向量索引"]
    CHUNK --> SOURCE["原文存储：文档、chunk、版本"]
    GRAPH -.->|来源引用| SOURCE
```

图、检索索引和原文存储是逻辑职责，不必分别部署三个数据库。属性图可使用 Neo4j，RDF 图可使用 GraphDB；Amazon Neptune 支持属性图与 RDF。Neo4j 也可以在图数据库内维护全文索引和向量索引。

实体向量应表达名称、类型、别名和必要背景，例如“张三，研发部工程师，负责知识助手项目”。语义相似只用于召回可能的对象，不能直接证明实体身份相同。

## 生产环境扩展：大量实体下的对齐与检索设计

以下流程将当前“全量实体交给模型”改为“候选召回后消歧”。候选数量根据数据与评估确定；精确匹配存在同名对象时也需要消歧。

```mermaid
flowchart TD
    QUESTION["用户问题"] --> MENTION["识别实体提及、类型与上下文"]
    MENTION --> EXACT["查询名称和别名索引"]
    EXACT --> UNIQUE{"是否唯一且符合上下文？"}
    UNIQUE -->|是| ENTITY["确定图中的 entity_id"]
    UNIQUE -->|否| LEXICAL["关键词、模糊匹配召回候选"]
    UNIQUE -->|否| DENSE["用提及上下文检索实体简介向量"]
    LEXICAL --> MERGE["与已有同名候选合并、去重，保留类型与简介"]
    DENSE --> MERGE
    MERGE --> RANK["规则、Cross-Encoder 或大模型消歧"]
    RANK --> CONFIDENT{"是否能确认实体？"}
    CONFIDENT -->|能| ENTITY
    CONFIDENT -->|不能| ABSTAIN["返回未匹配或请求澄清，不强行对齐"]
    ENTITY --> QUERY["按关系、方向和范围查询图：有界邻域或指定路径"]
    QUERY --> EVIDENCE["取回相关事实与原文证据"]
    EVIDENCE --> FILTER["筛选、重排并控制上下文预算"]
    FILTER --> ANSWER["大模型基于证据生成带来源的回答"]
```

实体对齐同时发生在建图与查询阶段：建图时避免同一对象被拆成多个节点，查询时把用户的提及链接到已有实体 ID。LangChain 可负责接口编排；关系遍历由程序或数据库查询执行。

## 参考

- [Neo4j 图模型与 Cypher](https://neo4j.com/docs/getting-started/cypher/)。
- [Neo4j 全文与向量索引](https://neo4j.com/docs/cypher-manual/current/indexes/search-indexes/)。
- [Amazon Neptune 图模型与查询语言](https://docs.aws.amazon.com/neptune/latest/userguide/access-graph-queries.html)。
- [GraphDB 与 RDF](https://www.ontotext.com/knowledgehub/fundamentals/graphdb-fundamentals/)。
- [Bi-Encoder 候选召回与 Cross-Encoder 消歧论文](https://aclanthology.org/2020.emnlp-main.519/)。
