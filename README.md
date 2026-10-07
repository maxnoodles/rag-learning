# RAG 学习项目

用于学习 RAG（检索增强生成），记录知识、代码实践和学习笔记。当前只建立目录骨架，内容后续自行填充。

## 目录

```text
rag-learning/
├── docs/                      # 概念、学习路线与资料
├── notes/                     # 学习笔记
├── data/
│   └── sample/                # 可以提交的示例数据
└── 02-RAG/
    └── 01-simple_rag/         # 第一个 RAG 实践
        ├── rag/              # 解析、分块、向量、检索与生成代码
        ├── docs/             # 实践使用的知识库文档
        └── tests/            # 测试
```

空目录使用 `.gitkeep` 保存，填入内容后可删除。
私人资料可放在 `data/private/`；`.env`、私人资料、虚拟环境和生成索引已被 Git 忽略。

## 参考

[AI-Engineer-from-scrach / 02-RAG / 01-simple_rag](https://github.com/juliepy/AI-Engineer-from-scrach/tree/main/02-RAG/01-simple_rag)

## 同步到 GitHub

在本项目目录中执行：

```bash
git status
git add .
git commit -m "docs: update RAG learning notes"
git push
```

执行 `git add .` 前检查待提交内容。修改通过上述命令手动同步。
