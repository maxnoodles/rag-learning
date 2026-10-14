#!/usr/bin/env python3
"""旧版单轮 RRF + MMR 基线，6 个手选问题；不评估当前两轮 Cross-Encoder 流程。"""
import os
os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rag.embedder import Embedder
from rag.context_builder import build_context
from rag.retriever import retrieve
from rag.vector_store import fingerprint,load_current

CASES=[
    ('FAISS IndexFlatIP 中 Flat 和 IP 分别代表什么，是否需要训练索引？','chat-011'),
    ('多模态 embedding 查图片时，是先把图片描述成文字再检索吗？','chat-014'),
    ('GGUF 量化模型和原始模型有哪些区别？','chat-005'),
    ('怎么从人与 AI 的日常对话里提取训练信号？','chat-021'),
    ('使用合成数据训练的学生模型有没有可能超过老师模型？','chat-017'),
    ('强化学习的奖励必须由人逐次给出吗？自动奖励怎么保证目标正确？','chat-023'),
]


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,default=Path('/Users/noodles/Documents/Codex/2026-10-06/h/outputs/models/embeddinggemma-2'))
    args=parser.parse_args()
    identity,_=fingerprint(ROOT/'data/private/chat_samples',args.model_path,512,64)
    snapshot,reason=load_current(ROOT/'data/indexes',identity)
    if snapshot is None:
        raise RuntimeError('先运行 run.py 建库：'+reason)
    model=Embedder(args.model_path)
    rows=[]
    for query,expected in CASES:
        recalled,selected=retrieve(snapshot,model.query(query),20,4,.7,query=query)
        rank=next((i for i,h in enumerate(recalled,1)
                   if snapshot.documents[h.chunk.doc_id].metadata.get('pair_id')==expected),None)
        evidence,_,_=build_context(snapshot,selected)
        retained=any(s.get('pair_id')==expected for e in evidence for s in e.sources)
        row=dict(query=query,expected_pair_id=expected,rank=rank,hit_at_1=rank==1,
                 hit_at_3=rank is not None and rank<=3,final_context_contains_expected=retained)
        rows.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
    report=dict(method="single_round_rrf_mmr_baseline",case_count=len(rows),hit_at_1=sum(r['hit_at_1'] for r in rows)/len(rows),
                hit_at_3=sum(r['hit_at_3'] for r in rows)/len(rows),
                final_context_hit=sum(r['final_context_contains_expected'] for r in rows)/len(rows),
                cases=rows,scope='hand-selected 6-case functional retrieval check, not a benchmark')
    target=ROOT/'artifacts/retrieval_evaluation.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False))


if __name__=='__main__':
    main()
