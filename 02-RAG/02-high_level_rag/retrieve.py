"""从已有 GraphDocument 中选择实体并用 BFS 检索，无需重新建图。"""
import argparse
from pathlib import Path

from graph_rag.pipeline import load_graph_documents, save_json
from graph_rag.retriever import KnowledgeGraph, retrieve

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='question → Codex 选实体 → BFS → 子图与原文证据')
    parser.add_argument('--question', required=True)
    parser.add_argument('--graph-path', type=Path,
                        default=ROOT / 'artifacts/graph_build/graph_documents.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/retrieval/latest.json')
    parser.add_argument('--hops', type=int, default=2)
    parser.add_argument('--direction', choices=['out', 'in', 'both'], default='both')
    parser.add_argument('--max-nodes', type=int, default=30)
    parser.add_argument('--max-seeds', type=int, default=3)
    parser.add_argument('--model', help='默认沿用本机 Codex 配置')
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    if (not args.question.strip() or args.hops < 0 or args.timeout < 1
            or not 1 <= args.max_seeds <= args.max_nodes):
        parser.error('问题不能为空；hops >= 0；timeout >= 1；1 <= max-seeds <= max-nodes')
    if not args.graph_path.is_file():
        parser.error('找不到图文档，请先运行 run.sh 建图，或指定 --graph-path')
    if args.output.resolve() == args.graph_path.resolve():
        parser.error('output 不能覆盖输入图文档')
    args.output.unlink(missing_ok=True)
    graph = KnowledgeGraph(load_graph_documents(args.graph_path))
    print(f'已加载 {len(graph.nodes)} 个实体、{len(graph.edges)} 条关系；正在调用 Codex 选择起点', flush=True)
    result = retrieve(args.question, graph, model=args.model, timeout=args.timeout,
                      max_seeds=args.max_seeds, hops=args.hops,
                      direction=args.direction, max_nodes=args.max_nodes)
    save_json(args.output, result)
    print('起点：' + ('、'.join(f'{e["type"]}:{e["id"]}'
                            for e in result['seed_selection']['entities']) or '无匹配'))
    print('选择理由：' + result['seed_selection']['reason'])
    for layer in result['subgraph']['layers']:
        print(f'{layer["hop"]} 跳：' + '、'.join(e['id'] for e in layer['entities']))
    for rel in result['subgraph']['relationships']:
        print(f'{rel["source"]["id"]} --{rel["type"]}--> {rel["target"]["id"]}')
        for evidence in rel['evidence']:
            print(f'  [{evidence["chunk_id"]}] {evidence["quote"]}')
    if result['subgraph']['truncated']:
        print('已达到节点数上限，检索结果被截断。')
    print(f'检索产物：{args.output.resolve()}')


if __name__ == '__main__':
    main()
