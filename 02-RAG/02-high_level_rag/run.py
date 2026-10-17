"""运行本节的图文档构建流程。"""
import argparse
from pathlib import Path

from graph_rag.pipeline import (build_graph_documents, create_transformer,
                                load_documents, save_json, split_documents, summarize)

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='示例文档 → chunks → Codex → GraphDocuments')
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/sample')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/graph_build')
    parser.add_argument('--chunk-size', type=int, default=60)
    parser.add_argument('--chunk-overlap', type=int, default=10)
    parser.add_argument('--model', help='可选；默认沿用本机 Codex 配置')
    parser.add_argument('--timeout', type=int, default=180, help='单个 chunk 抽取超时秒数')
    parser.add_argument('--chunks-only', action='store_true', help='只切块，不调用 Codex')
    args = parser.parse_args()
    if not 0 <= args.chunk_overlap < args.chunk_size or args.timeout < 1:
        parser.error('要求 0 <= chunk-overlap < chunk-size，timeout >= 1')
    # 完整运行前移除旧的最终产物，防止失败后误读上次结果。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ('graph_documents.json', 'summary.json'):
        (args.output_dir / name).unlink(missing_ok=True)
    documents = load_documents(args.input_dir)
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
    save_json(args.output_dir / 'chunks.json', [chunk.model_dump(mode='json') for chunk in chunks])
    print(f'{len(documents)} 份文档 → {len(chunks)} 个 chunks', flush=True)
    if args.chunks_only:
        return
    transformer = create_transformer(args.model, args.timeout)
    graphs = build_graph_documents(chunks, transformer, args.output_dir)
    report = dict(documents=len(documents), **summarize(graphs),
                  chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
                  model=args.model or '本机 Codex 默认模型')
    save_json(args.output_dir / 'summary.json', report)
    print(f'完成：{report["unique_nodes"]} 个不同节点，{report["unique_relationships"]} 条不同关系')
    print(f'产物：{args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
