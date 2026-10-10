# 测试报告

> 环境更新（2026-10-10）：以下是原完成时的记录。当前环境已迁移到 `02-RAG/.venv`，本章所有小节共用；旧小节环境和外部依赖 `.pth` 已移除。当前运行方式见 README。

生成时间：2026-10-09 23:13:08 CST。结果：全部列明的最终检查通过。测试使用真实本地模型和真实 Codex 调用，单元测试中的合成向量仅用于检验边界行为。

## 环境

Apple Silicon macOS；Python 3.12.14；Google embeddinggemma-2 / MPS / float32 / 768 维；FAISS 1.15.1；Codex Python SDK 0.162.0。最终 run.sh 调用使用配套 CLI 0.162.0，未使用 mock LLM。

## 单元测试

命令：`.venv/bin/python -m unittest discover -s tests -v`。

**26 / 26 通过，测试执行耗时 0.113 秒**（不包含启动和依赖导入）。

覆盖 UTF-8/BOM/换行、TXT/Markdown、空文件、编码错误、真实来源元数据、Unicode 切分覆盖、token 上限、问答边界、过长问题、非法 overlap、向量维度/NaN/零向量/归一化、存盘恢复、指纹失效、损坏文件拒绝、模拟发布失败回退、删除文档的新版本、MMR 多样性与 λ=1、超过语料量的 top-k、重复 chunk、同组补全、上下文预算、不相邻范围、重复正文的多来源、无效/缺失/虚构引用。

## 真实流程与缓存

| 检查 | 观察结果 | 总耗时 |
| --- | --- | ---: |
| 新目录首次建库 | cache_hit=false；25 文档、46 chunks；文档 embedding 1 次 | 12.101s |
| 第二次运行 | 同一 generation；cache_hit=true；文档 embedding 0 次 | 3.651s |
| --rebuild | 新 generation；cache_hit=false；文档 embedding 1 次 | 10.38s |
| 切分参数变更 | 512/64 → 256/32 自动重建；128 chunks | 12.04s |
| 最终本地检查 | run.sh --retrieve-only；缓存命中，实际 query 编码与检索正常 | 3.584s |
| 最终完整入口 | run.sh；SDK 固定运行时生成带引用答案 | 43.233s |
| 证据不足 | 返回 insufficient，未编造订单编号与金额 | 99.594s |
| 环境一致性 | pip check：No broken requirements found | 通过 |

首次、缓存和重建测试使用 artifacts/test-indexes，避免干扰默认知识库；当前默认索引仍是 512/64 参数。配置变化测试不会偷偷修改默认参数。早期开发中发现的 OpenMP 初始化冲突、来源 title 字段重复、协议中的 userMessage 类型，以及首次测试的变量初始化问题均已修复；以上为最终成功结果。

## 真实检索检查

命令：`.venv/bin/python scripts/evaluate_retrieval.py`。6 个手选问题，预期证据人工根据原始历史回答指定。

| 编号 | Query | 预期问答 | 首个相关 chunk 排名 | 最终上下文 |
| --- | --- | --- | ---: | --- |
| 1 | FAISS IndexFlatIP 中 Flat 和 IP 分别代表什么，是否需要训练索引？ | chat-011 | 1 | 保留 |
| 2 | 多模态 embedding 查图片时，是先把图片描述成文字再检索吗？ | chat-014 | 1 | 保留 |
| 3 | GGUF 量化模型和原始模型有哪些区别？ | chat-005 | 1 | 保留 |
| 4 | 怎么从人与 AI 的日常对话里提取训练信号？ | chat-021 | 1 | 保留 |
| 5 | 使用合成数据训练的学生模型有没有可能超过老师模型？ | chat-017 | 1 | 保留 |
| 6 | 强化学习的奖励必须由人逐次给出吗？自动奖励怎么保证目标正确？ | chat-023 | 1 | 保留 |

Hit@1=100%，Hit@3=100%，最终上下文保留预期来源=100%。统计为按预期问答是否命中的 Hit 指标，不是所有相关文档的完整 Recall@k。只证明这 6 个问题的功能表现，不能外推到新语料、复杂多跳问题或正式 benchmark。

## Codex 端到端回答

问题：FAISS IndexFlatIP 中的 Flat 和 IP 分别是什么意思？为什么归一化后可以按余弦相似度检索？

最终回答：

根据给定记录：

- **Flat**：完整保存向量，查询时逐一比较，是穷举式的精确检索；不压缩向量，也不需要训练索引。[E1]
- **IP**：Inner Product，即内积；按内积分数排序，分数越高越靠前。[E1]
- **为什么归一化后能检索余弦相似度？** 文档向量和查询向量都经过 L2 归一化后，二者的内积就等于余弦相似度。因此，按内积从高到低检索，也就是按余弦相似度从高到低检索。[E1]

注意：`IndexFlatIP` **不会自动归一化**，文档向量和查询向量都需要归一化。[E1]

验证结果：状态 answered；引用列表为 ['E1']；E1 对应 chat-011 / 011.md；正文引用与列表一致；输出活动仅包含 userMessage、reasoning 和 agentMessage，没有额外工具活动。人工核对 Flat、IP、归一化及不自动归一化四项均可在对应历史回答中找到。

最终服务 usage：input=12918，output=356；证据块估算 2746 tokens。SDK 系统/协议开销使服务计数高于本地证据估算。

## 证据不足验证

问题：这份聊天记录有没有说明我的火星旅行订单编号和付款金额？

实际回答：给定聊天片段没有提及你的火星旅行订单编号或付款金额，因此无法从这些证据确定这两项信息。[E1][E2][E3][E4]

状态 insufficient；引用编号均存在。该检查使用同一 Python SDK 和应用内 CLI 0.162.0-alpha.2；最终完整入口另外验证了配套稳定 CLI 0.162.0。没有宣称检索不到就能证明全库没有，只说明给定检索片段无法确定。

## 记录与范围

完整原文输出、prompt、模型结果及阶段耗时保存于 Git 忽略的 artifacts/：fresh_build、cache_hit、forced_rebuild、config_change、launcher_end_to_end、insufficient_evidence、final_local_check、retrieval_evaluation 的 JSON/日志，以及 unit_tests.log、pip_check.log、environment.json。正常使用直接在终端打印，无需逐个查文件。

未测试正式大规模 benchmark、多进程压力、断电恢复、恶意文档的完整攻击测试或每个事实的自动语义引用验证。已做模板中指令/数据区分、禁用生成阶段外部工具、引用编号校验；这些措施不能视作完整的 prompt injection 防护证明。
