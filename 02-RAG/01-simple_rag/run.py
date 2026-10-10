#!/usr/bin/env python3
"""打印 RAG 关键结果；后续自动复用索引，--rebuild 强制重建。"""
import argparse
import fcntl
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'

from rag.chunker import make_chunks
from rag.codex_adapter import generate
from rag.context_builder import build_context, make_prompt, token_count
from rag.document_parser import parse_documents
from rag.embedder import Embedder
from rag.retriever import retrieve
from rag.vector_store import fingerprint, load_current, save_snapshot

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = Path('/Users/noodles/Documents/Codex/2026-10-06/h/outputs/models/embeddinggemma-2')
DEFAULT_QUERY = 'FAISS IndexFlatIP 中的 Flat 和 IP 分别是什么意思？为什么归一化后可以按余弦相似度检索？'


def execute(args):
    report = {'query':args.query,'status':'running'}
    artifact = args.output_json
    print(f'问题：{args.query}',flush=True)
    try:
        if artifact:
            artifact.parent.mkdir(parents=True,exist_ok=True)
        identity, config = fingerprint(args.docs,args.model_path,args.chunk_tokens,args.overlap_tokens)
        args.index_dir.mkdir(parents=True,exist_ok=True)
        with (args.index_dir/'build.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                snapshot, reason = load_current(args.index_dir,identity)
                if args.rebuild:
                    snapshot, reason = None,'指定了 --rebuild，强制重建'
                report['cache_hit'] = snapshot is not None
                report['cache_reason'] = reason
                print(('缓存命中：' if snapshot else '需要建库：')+reason,flush=True)
                embedder = Embedder(args.model_path,args.device)
                print(f'Embedding：设备 {embedder.device}；{embedder.dimension} 维',flush=True)
                if snapshot is None:
                    documents = parse_documents(args.docs)
                    chunks = make_chunks(documents,embedder.tokenizer,args.chunk_tokens,args.overlap_tokens)
                    print(f'建库：{len(documents)} 文档 → {len(chunks)} chunks',flush=True)
                    vectors = embedder.documents(chunks)
                    report['document_embedding_calls'] = 1
                    # 建库中源文件变化时保留上一快照，避免发布混合版本。
                    after, _ = fingerprint(args.docs,args.model_path,args.chunk_tokens,args.overlap_tokens)
                    if after != identity:
                        raise RuntimeError('建库期间文档或模型发生变化，未发布；请重新运行')
                    snapshot = save_snapshot(args.index_dir,identity,config,documents,chunks,vectors)
                    print('索引已保存',flush=True)
                else:
                    report['document_embedding_calls'] = 0
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN)
        report.update(generation=snapshot.generation,document_count=len(snapshot.documents),chunk_count=len(snapshot.chunks))
        query_vector = embedder.query(args.query)
        recalled, selected = retrieve(snapshot,query_vector,args.fetch_k,args.top_k,args.mmr_lambda)
        print(f'检索：召回 {len(recalled)} 个候选 → MMR 选择 {len(selected)} 个片段',flush=True)
        for rank,hit in enumerate(selected,1):
            doc = snapshot.documents[hit.chunk.doc_id]
            print(f'  #{rank} {doc.source} chunk={hit.chunk.chunk_id} | relevance={hit.relevance:.6f} '
                  f'redundancy={hit.redundancy:.6f} MMR={hit.mmr_score:.6f}',flush=True)
        evidence, context, events = build_context(snapshot,selected,args.context_tokens)
        print(f'上下文：{len(evidence)} 组证据；预算 {args.context_tokens} tokens',flush=True)
        for event in events:
            print(f'  {event}',flush=True)
        print(context,flush=True)
        prompt = make_prompt(args.query,context) if artifact or (evidence and not args.retrieve_only) else None
        # 详细序列化和展示用计数只在请求保存报告时执行。
        if artifact:
            report.update(recalled=[asdict(h) for h in recalled],
                          selected=[asdict(h) for h in selected],
                          evidence=[asdict(e) for e in evidence],
                          context_tokens_estimated=token_count(context),prompt=prompt)
        if args.retrieve_only:
            report['answer'] = None
            print('--retrieve-only：检索完成',flush=True)
        elif not evidence:
            report['answer'] = {'status':'insufficient','answer':'上下文预算内没有可用证据，无法回答。','citations':[]}
            print(report['answer']['answer'],flush=True)
        else:
            print('Codex 正在生成回答…',flush=True)
            report['answer'] = generate(prompt,{e.citation_id for e in evidence},args.codex_model,
                                        args.timeout,args.codex_bin)
            print(report['answer']['answer'],flush=True)
            print(f'回答状态：{report["answer"]["status"]}；来源：{report["answer"]["citations"]}',flush=True)
        report['status'] = 'completed'
        return report
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        print(f'执行失败：{report["error"]}',file=sys.stderr,flush=True)
        raise
    finally:
        if artifact:
            artifact.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query',default=DEFAULT_QUERY)
    parser.add_argument('--docs',type=Path,default=ROOT/'data/private/chat_samples')
    parser.add_argument('--model-path',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--index-dir',type=Path,default=ROOT/'data/indexes')
    parser.add_argument('--rebuild',action='store_true')
    parser.add_argument('--retrieve-only',action='store_true')
    parser.add_argument('--chunk-tokens',type=int,default=512)
    parser.add_argument('--overlap-tokens',type=int,default=64)
    parser.add_argument('--fetch-k',type=int,default=20)
    parser.add_argument('--top-k',type=int,default=4)
    parser.add_argument('--mmr-lambda',type=float,default=0.7)
    parser.add_argument('--context-tokens',type=int,default=6000)
    parser.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    parser.add_argument('--codex-model',default=None,help='省略时使用 Codex 当前默认模型')
    parser.add_argument('--codex-bin',default=None)
    parser.add_argument('--timeout',type=int,default=300)
    parser.add_argument('--output-json',type=Path,default=None,help='可选，保存详细检索结果、证据和 prompt')
    args = parser.parse_args()
    if not args.query.strip() or not 1 <= args.top_k <= args.fetch_k or not 0 <= args.mmr_lambda <= 1:
        parser.error('query 不能为空；1 <= top-k <= fetch-k；0 <= mmr-lambda <= 1')
    if not 32 <= args.chunk_tokens <= 8192 or not 0 <= args.overlap_tokens < args.chunk_tokens:
        parser.error('32 <= chunk-tokens <= 8192；0 <= overlap-tokens < chunk-tokens')
    if args.context_tokens < 64 or args.timeout <= 0:
        parser.error('context-tokens 至少 64，timeout 必须为正数')
    args.rebuild = True
    args.docs=args.docs.resolve(); args.model_path=args.model_path.resolve(); args.index_dir=args.index_dir.resolve()
    try:
        execute(args)
    except Exception:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
