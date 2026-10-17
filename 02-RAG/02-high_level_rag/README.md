# Graph RAG：图文档构建与 BFS 检索

本节专注 Graph RAG。目前完成图文档构建，以及 question → Codex 选择已有实体 → BFS 子图检索。

当前实现与生产环境扩展设计见 [Graph RAG 流程图](docs/GRAPH_RAG_FLOW.md)。

## 示例数据

`data/sample/` 中的 5 份中文文档完全虚构，围绕星河科技的 4 名成员、2 个部门、3 个项目和 3 种技术展开。句子尽量直接，方便逐条核对模型输出。按本节 schema，共有 13 个实体和 30 条不同的直接关系。

| 文件 | 内容 |
| --- | --- |
| `01_team.txt` | 部门归属、人员部门、项目负责人及参与者 |
| `02_knowledge_assistant.txt` | 知识助手项目的归属、维护部门、技术和依赖 |
| `03_customer_portal.txt` | 客户门户项目的归属、维护部门、技术和依赖 |
| `04_dashboard.txt` | 数据看板项目的归属、维护部门和技术 |
| `05_technology.txt` | 技术背景及成员熟悉的技术 |

例如，后续可通过跨文档路径回答“张三负责的项目使用什么技术”：

```text
张三 ──LEADS──→ 知识助手项目 ──USES──→ Python
                              └──USES──→ Redis
```

检索可以从张三出发找到这条路径及其原文证据。

## 运行

使用章节共享的 `../.venv`。从 `02-RAG` 目录安装和运行：

```sh
.venv/bin/python -m pip install -r requirements.txt
sh 02-high_level_rag/run.sh
```

抽取使用已登录的本地 Codex，通过现有官方 `openai-codex` Python SDK 调用，沿用本机默认模型；可用 `--model` 显式选择。每个 chunk 同步调用一次，默认超时 180 秒，消耗 Codex 使用额度。单元测试不调用模型。

只看切块结果，或用小预算观察一份文档被拆成多个 chunk：

```sh
sh 02-high_level_rag/run.sh --chunks-only --chunk-size 60 --chunk-overlap 10 --output-dir 02-high_level_rag/artifacts/chunk_demo
```

默认输入为 `data/sample/`，也可用 `--input-dir` 指定自己的 `.txt` 文件目录。

建图完成后，直接检索已有图文档，每次问题只调用一次 Codex 选择起点，无需重新抽取整批数据：

```sh
sh 02-high_level_rag/retrieve.sh --question '张三负责的项目用了哪些技术？'
sh 02-high_level_rag/retrieve.sh --question '哪些项目使用Redis？' --hops 1 --direction in
sh 02-high_level_rag/retrieve.sh --question '张三和李四分别负责哪些项目？'
```

默认输出到 `artifacts/retrieval/latest.json`。用 `--output` 可保存多个问题的结果，`--graph-path` 可指定图文档路径。

## 实体选择与 BFS 检索

`graph_rag/retriever.py` 将图文档合并成内存图：节点按 `(type, id)` 区分，边按有向三元组合并，并保留每个来源 chunk 和原文引文。孤立节点也可作为候选。

1. 将所有已有实体的类型、名称和候选号提供给 Codex。模型只选择候选号，支持多个起点，最多 3 个；无法匹配时返回空数组。返回值还会校验，候选以外的实体会报错。
2. 用 `deque` 执行多源 BFS，默认最多 2 跳、30 个节点。`out` 沿出边，`in` 沿入边，`both` 双向；默认 `both`，便于反向查找负责人、技术使用者等。
3. 在 BFS 找到的节点集合中保留全部原始有向边，构成诱导子图，而不只是搜索树边。方向参数控制发现哪些节点；输出中的事实方向不变。例如反向检索仍返回“项目 --USES--> Redis”。即使某个节点已访问，也保留指向它的其他证据边。
4. 返回起点、选择理由、按最小跳数分层的节点、三元组、每条边的 evidence 以及原文 Document。达到节点上限时 `truncated=true`，不隐瞒截断。

当前是基于距离的候选子图召回，没有按问题筛选关系、重排、向量检索或最终答案生成。特别是双向扩展可能带来无关邻居；子图中有一条事实，不等于它能回答当前问题。多起点 BFS 返回的是到任一起点的最小距离邻域，并不直接计算共同参与者或两实体的交集。

所有实体都交给模型适合当前 13 个节点的小图；图变大后需先召回少量候选再消歧。当前也未建立人工别名表，简称由模型在给定候选中判断。

## 三个步骤

1. **递归切块**：`RecursiveCharacterTextSplitter.from_tiktoken_encoder()` 按段落 → 换行 → 中文句末 → 空格 → 字符逐级尝试；默认预算为 60 tokens、重叠上限为 10 tokens。以 `o200k_base` 计数，作为明确的切块度量，不表示默认模型必然使用同一个 tokenizer。短文档可以整个保留为一个 chunk，实际重叠也可能小于设定值。首次使用 tiktoken 词表需要下载，之后使用缓存。
2. **抽取实体与关系**：`CodexChatModel.with_structured_output()` 将 LangChain 的抽取 schema 传给 Codex，校验结构化结果，再交给 `LLMGraphTransformer`。只根据当前 chunk 抽取明确陈述，关系附带 `evidence` 原文引文。
3. **构建图文档**：真实调用 `LLMGraphTransformer.convert_to_graph_documents(chunks)`。返回值已经是 `List[GraphDocument]`；每个对象包含 `nodes`、`relationships` 和 `source: Document`，无需再次手工包装同名对象。

上述组件属于 **LangChain 生态**，不是 LangGraph 的切分 API；本节使用当前 `langchain-neo4j` 包中的转换器。知识图谱与 LangGraph 工作流中的流程图也是不同概念。

节点类型限定为 `Person`、`Company`、`Department`、`Project`、`Technology`。关系同时约束类型和方向：

| 起点 | 关系 | 终点 |
| --- | --- | --- |
| Department | PART_OF | Company |
| Person | WORKS_IN | Department |
| Person | LEADS / PARTICIPATES_IN | Project |
| Person | KNOWS | Technology |
| Project | OWNED_BY | Company |
| Project | MAINTAINED_BY | Department |
| Project | USES | Technology |
| Project | DEPENDS_ON | Project |

## 输出与代码入口

完整运行产物位于 `artifacts/graph_build/`，由 Git 忽略：

- `chunks.json`：chunk 的原文、源文件、稳定 ID、顺序和 token 数。
- `graph_documents.json`：每个 chunk 对应的 GraphDocument，包含节点、带原文引文的关系和源 chunk。
- `summary.json`：文档数、图文档数、跨文档去重后的节点/关系数量和参数。去重只用于统计，原始图文档保留各自来源。

2026-10-15 真实 Codex 验证：5 份文档 → 6 个 chunks → 6 个 GraphDocument，得到 13 个不同节点、30 条不同关系。与样本预期的 30 条直接关系逐项比对，无遗漏或多余关系，来源往返载入和原文引文检查通过；此次核验结果保存在 `artifacts/graph_build/validation.json`。新增 5 项离线测试与基础版 57 项回归测试通过，共享环境 `pip check` 通过。这是小样本功能验证，不是通用抽取准确率评测。

构建开始时移除旧的最终产物，全部 chunk 成功后才写入最终图文档；失败时不会把上次的最终结果当作本次结果。

从 `run.py` 开始读；流程在 `graph_rag/pipeline.py`，Codex 适配在 `graph_rag/codex_model.py`。用 `load_graph_documents(path)` 可以重新载入真正的 GraphDocument 对象。

检索从 `retrieve.py` 开始读，实体选择、邻接表和 BFS 在 `graph_rag/retriever.py`。检索报告中的 `sources` 通过 `metadata.chunk_id` 与关系的 `evidence` 对应。

## 校验与边界

落盘前检查节点是否出现在原文、关系端点是否存在、关系方向是否符合 schema、引文是否为原文连续子串且包含两端实体。这不能证明模型已经抽全关系或完全理解语义，仍需对照原文核验。LangChain 的结构化转换会对节点 ID 做 title-case 规范化，中文不受影响；真实语料的实体别名合并留待后续。

在本小节目录运行离线测试：

```sh
../.venv/bin/python -m unittest discover -s tests -v
```

2026-10-15 检索验证：本小节共 19 项离线测试通过，基础版 57 项回归测试通过。真实 Codex 对普通查询、反向查询、多实体查询和无匹配查询的 4 个样例通过，产物分别在 `artifacts/retrieval/forward.json`、`reverse.json`、`multiple.json`、`no_match.json`。这验证了预期起点、必要关系及引文保留，不是最终回答准确率评测。

与用户提供的教学脚本的比较见 [实现对照](docs/REFERENCE_COMPARISON.md)。

接口参考：[LangChain token 切块](https://docs.langchain.com/oss/python/integrations/splitters/split_by_token)、[convert_to_graph_documents](https://reference.langchain.com/python/langchain-neo4j/graph_transformers/llm/LLMGraphTransformer/convert_to_graph_documents)、[OpenAI 官方 Codex 结构化输出说明](https://developers.openai.com/codex/noninteractive/)。
