# 完成报告

> 2026-10-15 更新：当前入口已改为原问题与 Codex 抽象问题两轮混合检索，每轮 top_k=4，Merge 去重后由本地 BGE Cross-Encoder 取 top_n=2。45 个测试及真实 MPS 检索、重排、Codex 回答验证通过。当前用法见 README，最新验证见 TEST_REPORT；下文为早期版本的历史完成记录。


> 环境更新（2026-10-10）：以下是原完成时的记录。当前环境已迁移到 `02-RAG/.venv`，本章所有小节共用；旧小节环境和外部依赖 `.pth` 已移除。当前运行方式见 README。

生成时间：2026-10-09 23:13:08 CST。项目：`/Users/noodles/Desktop/ai-engineer-learning/rag-learning/02-RAG/01-simple_rag`。

## 完成结论

已实现并验证可观察的本地 RAG 学习流程。25 组真实历史问答 → 46 个默认 chunk → 本地 Google 向量 → FAISS/SQLite 磁盘快照 → 原始 query 检索 → MMR → 精确去重与同组补全 → 完整上下文打印 → Codex 基于证据回答。

## 已交付功能

- TXT / Markdown parser 注册表，UTF-8/BOM、换行规范化和明确的空文档/编码错误。
- 每个样例文件一组问答，长问答保留问题后分片；字符范围对应持久化正文，切分计数包含标题和前缀。
- 本地 `google/embeddinggemma-2`，MPS、float32、768 维、L2 归一化；查询使用 SearchQuery 前缀。
- FAISS IndexFlatIP + SQLite ID/正文映射，manifest 完整性校验、单机构建锁与原子发布。
- 首次建库，后续校验命中则文档 embedding 调用为 0；`--rebuild` 强制重建。
- 默认候选 20、MMR 最多选 4、λ=0.7，打印每个选择的 relevance / redundancy / MMR。
- 精确去重，优先补全同组问答；超预算时只合并同文档的重叠/相邻命中范围，保留来源。
- 完整打印 prompt、召回原文与分数、各阶段耗时，使用终端直接观察。
- 官方 Codex Python SDK + 配套 CLI 0.162.0，复用已登录账号。生成关闭 MCP、shell、web 等工具，检查返回活动与引用编号。仅回答证据支持的内容。
- 运行脚本、参数、26 项单元测试、真实检索检查、完成和测试报告。

## 运行

```sh
cd /Users/noodles/Desktop/ai-engineer-learning/rag-learning/02-RAG/01-simple_rag
./run.sh
./run.sh --query '强化学习的奖励必须由人类每次给出吗？'
./run.sh --rebuild
./run.sh --retrieve-only
```

程序不要求逐个查文件。可选 `--output-json artifacts/my-run.json` 用于保存记录。更多参数见 README.md。

## 样例来源

全部是原始 Codex 用户问题与助手最终回答，覆盖 5 个聊天：

| 原始聊天标题 | 问答数量 |
| --- | ---: |
| 查找适合本机的轻量嵌入模型 | 14 |
| 理解合成数据与过拟合 | 4 |
| 探索企业文化内化到模型的途径 | 3 |
| 解释强化学习 | 2 |
| 设计不停机的索引切换 | 2 |

保留前一条用户问题、角色、标题、Asia/Shanghai 时间、thread ID 与 JSONL 原文行号；不导入系统/开发者指令、工具输出、内部推理、过程更新或机器引用尾注。样例在 `data/private/chat_samples/`，来源清单在该目录的 corpus_manifest.json。没有加入当前实现聊天作为自证答案。

## 模型、依赖与下载透明说明

模型直接复用 `/Users/noodles/Documents/Codex/2026-10-06/h/outputs/models/embeddinggemma-2`；没有重新下载权重。当前完整权重 SHA256：`197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`。本地下载清单 revision 为 `914f7f89142e33e77833254d9c9b90c3cef7303b`。

既有大型 embedding 依赖通过项目 `.venv` 中的 local_gemma_dependencies.pth 复用，不改写原环境。路径：`/Users/noodles/Documents/Codex/2026-10-06/h/work/gemma-venv/lib/python3.12/site-packages`。

| 复用依赖 | 版本 | 安装位置 |
| --- | --- | --- |
| torch | 2.14.1 | 既有 gemma-venv |
| transformers | 5.19.0 | 既有 gemma-venv |
| sentence-transformers | 6.1.0 | 既有 gemma-venv |
| faiss-cpu | 1.15.1 | 既有 gemma-venv |

本次新增依赖来自官方 PyPI，安装在 `/Users/noodles/Desktop/ai-engineer-learning/rag-learning/02-RAG/01-simple_rag/.venv/lib/python3.12/site-packages`：

| 新增依赖 | 版本 | 安装位置 |
| --- | --- | --- |
| openai-codex | 0.162.0 | 项目 .venv |
| openai-codex-cli-bin | 0.162.0 | 项目 .venv |
| tiktoken | 0.14.0 | 项目 .venv |
| pydantic | 2.14.0 | 项目 .venv |
| pydantic-core | 2.50.0 | 项目 .venv |
| annotated-types | 0.8.0 | 项目 .venv |
| typing-inspection | 0.4.4 | 项目 .venv |
| requests | 2.34.2 | 项目 .venv |
| charset-normalizer | 3.5.2 | 项目 .venv |
| urllib3 | 2.8.0 | 项目 .venv |

SDK 用于答案生成；配套 CLI 是固定版本运行时；tiktoken 用于生成侧预算估算；pydantic 等是 SDK 的结构化协议依赖，requests 等支持 token 表加载。CLI ARM64 wheel 约 124 MB，官方 PyPI SHA256 校验通过；校验记录在 artifacts/runtime-download/provenance.json。首用的 o200k_base token 表来自 OpenAI 公共 encoding 资源，运行脚本使用 artifacts/tiktoken-cache/ 缓存。

安装期间镜像未提供 SDK，改用官方 PyPI；大 wheel 下载超时后使用当前 macOS HTTP 代理 127.0.0.1:7892 下载官方文件，校验哈希再安装。未改动用户的系统代理、Codex 登录或全局配置。最终 pip check 无依赖冲突。

## 限制与后续学习方向

- 仅支持 TXT/Markdown、纯向量检索和原始 query；PDF/OCR、全文检索、查询改写、reranker、MCP 入口和元数据过滤未实现。
- MMR 的多样性可能引入较弱相关片段；它不保证事实正确，也不代替拒答和检索评估。
- `context-tokens=6000` 是证据预算，o200k_base 是估算计数。SDK 有额外系统/协议开销；最终完整调用实际 input tokens 为 12918，不能把证据预算当作整个请求上限。
- 引用校验确认编号存在且正文/列表一致，不能自动证明每句话都被引用支持。历史助手回答也不代表当前事实。
- 现有 .venv 通过 .pth 依赖旧 embedding 环境；搬迁或删除该环境后应按 requirements.txt 重建。
- PyTorch/FAISS 在本机存在 OpenMP 共存边界，已固定导入/初始化顺序和单线程并跑通真实流程。更换依赖后需要重新验证。
- 保留旧 generation 以便回退，暂未实现自动清理；测试了模拟发布失败保留旧快照，没有做断电恢复或并发压力测试。
- 聊天正文、索引和运行记录已被 Git 忽略；未提交或推送。调用 Codex 会发送筛选后的上下文，这是已确认的生成方案。

参考：[Google 模型说明](https://huggingface.co/google/embeddinggemma-2)、[Codex SDK](https://learn.chatgpt.com/docs/codex-sdk)、[FAISS 索引](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)。
