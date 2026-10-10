#!/usr/bin/env python3
"""运行一次即可观察 RAG 每一步；后续自动复用索引，--rebuild 强制重建。"""
import argparse
import contextlib
import fcntl
import json
import os
import sys
import time
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


@contextlib.contextmanager
def stage(name, timings):
    print(f'\n{"="*24} {name} {"="*24}',flush=True)
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round(time.perf_counter()-start,3)
        print(f'耗时：{timings[name]:.3f}s',flush=True)


def execute(args):
    started = time.perf_counter()
    timings = {}
    report = {'query':args.query,'status':'running','timings':timings}
    artifact = args.output_json
    if artifact:
        artifact.parent.mkdir(parents=True,exist_ok=True)
    print('本地 RAG：TXT/Markdown → 问答 chunk → Google embedding → FAISS → MMR → Codex',flush=True)
    try:
        with stage('1. 缓存检查',timings):
            identity, config = fingerprint(args.docs,args.model_path,args.chunk_tokens,args.overlap_tokens)
            args.index_dir.mkdir(parents=True,exist_ok=True)
            lock = (args.index_dir/'build.lock').open('a')
            fcntl.flock(lock,fcntl.LOCK_EX)
            snapshot, reason = load_current(args.index_dir,identity)
            if args.rebuild:
                snapshot, reason = None,'指定了 --rebuild，强制重新解析、切分和 embedding'
            report['cache_hit'] = snapshot is not None
            report['cache_reason'] = reason
            print(('缓存命中：' if snapshot else '需要建库：')+reason,flush=True)
        try:
            with stage('2. 加载本地 embedding 模型',timings):
                embedder = Embedder(args.model_path,args.device)
                print(f'{args.model_path}\n设备：{embedder.device}；768 维；float32；归一化')
                # 模型 API 的超大 tokenizer sentinel 不能当作实际上下文限制。
                if args.chunk_tokens > 8192:
                    raise ValueError('学习版 chunk tokens 上限为 8192，避免无界输入')
            if snapshot is None:
                with stage('3. Parser 解析',timings):
                    documents = parse_documents(args.docs)
                    for d in documents:
                        print(f'\n[{d.source}] {d.title} | {len(d.text)} 字符 | {d.metadata.get("timestamp", "无时间信息")}')
                        print(d.text)
                with stage('4. Chunker 按问答切分',timings):
                    chunks = make_chunks(documents,embedder.tokenizer,args.chunk_tokens,args.overlap_tokens)
                    for c in chunks:
                        print(f'\nchunk={c.chunk_id} doc={c.doc_id} #{c.ordinal} 范围=[{c.start},{c.end}) tokens={c.token_count}')
                        print(c.text)
                with stage('5. 文档 embedding',timings):
                    vectors = embedder.documents(chunks)
                    report['document_embedding_calls'] = 1
                    print(f'向量矩阵：{vectors.shape}；float32；每个向量 L2 范数约为 1')
                    print(f'第一条向量前 8 维：{vectors[0,:8].tolist()}（不打印全部 768 维）')
                with stage('6. FAISS + SQLite 持久化',timings):
                    # 建库中源文件变化时保留上一快照，避免发布混合版本。
                    after, _ = fingerprint(args.docs,args.model_path,args.chunk_tokens,args.overlap_tokens)
                    if after != identity:
                        raise RuntimeError('建库期间文档或模型发生变化，未发布；请重新运行')
                    snapshot = save_snapshot(args.index_dir,identity,config,documents,chunks,vectors)
                    print(f'generation={snapshot.generation}；{len(documents)} 文档；{len(chunks)} chunks')
            else:
                report['document_embedding_calls'] = 0
                print('\n[3–6 跳过] 从磁盘恢复正文、chunk 和文档向量；本次仅计算 query 向量。')
                for c in snapshot.chunks:
                    print(f'缓存 chunk={c.chunk_id} 范围=[{c.start},{c.end}) tokens={c.token_count}')
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN); lock.close()
        report.update(generation=snapshot.generation,document_count=len(snapshot.documents),chunk_count=len(snapshot.chunks))
        with stage('7. 原始 query embedding',timings):
            print(args.query)
            query_vector = embedder.query(args.query)
            print(f'维度：{query_vector.shape}；SearchQuery 前缀；没有 query 改写')
        with stage('8. Retriever 候选召回',timings):
            recalled, selected = retrieve(snapshot,query_vector,args.fetch_k,args.top_k,args.mmr_lambda)
            for rank,hit in enumerate(recalled,1):
                doc = snapshot.documents[hit.chunk.doc_id]
                print(f'\n#{rank} cosine={hit.relevance:.6f} source={doc.source} chunk={hit.chunk.chunk_id}')
                print(hit.chunk.text)
            report['recalled'] = [asdict(h) for h in recalled]
        with stage('9. MMR 去冗余选择',timings):
            print(f'Maximal Marginal Relevance；lambda={args.mmr_lambda}；最多选择 {args.top_k} 个 chunk')
            print('score = lambda × relevance − (1−lambda) × redundancy；首项选最相关 chunk')
            for rank,hit in enumerate(selected,1):
                print(f'选择 #{rank} {hit.chunk.chunk_id} | relevance={hit.relevance:.6f} '
                      f'redundancy={hit.redundancy:.6f} MMR={hit.mmr_score:.6f}')
            report['selected'] = [asdict(h) for h in selected]
        with stage('10. 精确去重、同组问答补全与上下文预算',timings):
            evidence, context, events = build_context(snapshot,selected,args.context_tokens)
            for event in events:
                print(event)
            print(f'{len(selected)} chunks → {len(evidence)} 组证据；'
                  f'o200k_base 估算 {token_count(context)} tokens / 预算 {args.context_tokens}')
            report['evidence'] = [asdict(e) for e in evidence]
            report['context_tokens_estimated'] = token_count(context)
        with stage('11. 完整生成上下文',timings):
            prompt = make_prompt(args.query,context)
            print(prompt,flush=True)
            print(f'整个 prompt 估算 {token_count(prompt)} tokens；计数器与 embedding tokenizer 独立。')
            report['prompt'] = prompt
        with stage('12. Codex 基于证据生成回答',timings):
            if args.retrieve_only:
                report['answer'] = None
                print('--retrieve-only：仅验证检索，没有调用 Codex。')
            elif not evidence:
                report['answer'] = {'status':'insufficient','answer':'上下文预算内没有可用证据，无法回答。','citations':[]}
                print(report['answer']['answer'])
            else:
                report['answer'] = generate(prompt,{e.citation_id for e in evidence},args.codex_model,
                                            args.timeout,args.codex_bin)
                print(report['answer']['answer'],flush=True)
                print(f'回答状态：{report["answer"]["status"]}；来源：{report["answer"]["citations"]}')
            report['status'] = 'completed'
        return report
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        print(f'\n执行失败：{report["error"]}',file=sys.stderr,flush=True)
        raise
    finally:
        report['total_seconds'] = round(time.perf_counter()-started,3)
        print(f'\n总耗时：{report["total_seconds"]:.3f}s',flush=True)
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
    parser.add_argument('--output-json',type=Path,default=None,help='可选，保存验证报告；终端仍打印全部阶段')
    args = parser.parse_args()
    if not args.query.strip() or not 1 <= args.top_k <= args.fetch_k or not 0 <= args.mmr_lambda <= 1:
        parser.error('query 不能为空；1 <= top-k <= fetch-k；0 <= mmr-lambda <= 1')
    if not 32 <= args.chunk_tokens <= 8192 or not 0 <= args.overlap_tokens < args.chunk_tokens:
        parser.error('32 <= chunk-tokens <= 8192；0 <= overlap-tokens < chunk-tokens')
    if args.context_tokens < 64 or args.timeout <= 0:
        parser.error('context-tokens 至少 64，timeout 必须为正数')
    args.docs=args.docs.resolve(); args.model_path=args.model_path.resolve(); args.index_dir=args.index_dir.resolve()
    try:
        execute(args)
    except Exception as exc:
        print(f'程序错误：{type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
