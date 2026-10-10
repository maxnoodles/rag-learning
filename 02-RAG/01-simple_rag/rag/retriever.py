"""先召回候选，再做 MMR；分数不是正确答案概率。"""
import numpy as np

from .schemas import Hit


def retrieve(snapshot, query_vector, fetch_k=20, top_k=4, mmr_lambda=0.7):
    if not 0 <= mmr_lambda <= 1 or not 1 <= top_k <= fetch_k:
        raise ValueError('MMR lambda 必须在 [0,1]；要求 1 <= top_k <= fetch_k')
    scores, ids = snapshot.index.search(query_vector.reshape(1,-1), min(fetch_k, len(snapshot.chunks)))
    candidates = [(int(i),float(score)) for i,score in zip(ids[0],scores[0]) if i >= 0]
    recalled = [Hit(snapshot.chunks[i],score) for i,score in candidates]
    selected, remaining, chosen_ids = [], list(candidates), []
    while remaining and len(selected) < top_k:
        ranked = []
        for i,relevance in remaining:
            redundancy = max(float(snapshot.vectors[i] @ snapshot.vectors[j]) for j in chosen_ids) if chosen_ids else 0.0
            score = mmr_lambda*relevance-(1-mmr_lambda)*redundancy
            ranked.append((score,relevance,-i,i,redundancy))
        # 第一项选最相关者；后续项才考虑多样性，避免 lambda=0 首项无意义。
        best = max(ranked, key=lambda r:(r[1],r[2])) if not chosen_ids else max(ranked)
        score,relevance,_,i,redundancy = best
        selected.append(Hit(snapshot.chunks[i],relevance,redundancy,score))
        chosen_ids.append(i)
        remaining = [(j,s) for j,s in remaining if j != i]
    return recalled, selected
