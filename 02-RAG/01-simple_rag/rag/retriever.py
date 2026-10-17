"""FAISS + BM25 → 加权 RRF 召回；合并后由 MMR 压缩候选。"""
from dataclasses import replace

import numpy as np

from .schemas import Hit

def recall(snapshot, query_vector, fetch_k=20, top_k=4, *,
           query=None, vector_weight=0.5, rrf_k=60):
    if not 1 <= top_k <= fetch_k:
        raise ValueError('要求 1 <= top_k <= fetch_k')
    if not 0 <= vector_weight <= 1 or rrf_k < 1:
        raise ValueError('vector weight 必须在 [0,1]；RRF k 至少为 1')
    if vector_weight < 1 and (not query or not query.strip()):
        raise ValueError('BM25 混合检索需要非空 query 原文')
    if not snapshot.chunks:
        return []
    vector_results = []
    if vector_weight > 0:
        scores, ids = snapshot.index.search(query_vector.reshape(1,-1), min(fetch_k, len(snapshot.chunks)))
        vector_results = [(int(i),float(s)) for i,s in zip(ids[0],scores[0]) if i >= 0]
    lexical_results = snapshot.bm25.search(query, fetch_k) if vector_weight < 1 else []
    hits = {}
    for rank, (i, score) in enumerate(vector_results, 1):
        hits[i] = Hit(snapshot.chunks[i], 0, vector_score=score, vector_rank=rank,
                      rrf_score=vector_weight/(rrf_k+rank))
    for rank, (i, score) in enumerate(lexical_results, 1):
        hit = hits.setdefault(i, Hit(snapshot.chunks[i], 0))
        hit.bm25_score, hit.bm25_rank = score, rank
        hit.rrf_score += (1-vector_weight)/(rrf_k+rank)
    # 未命中词时按向量通道独立归一化，避免 MMR relevance 被空通道压低。
    active_weight = (vector_weight if vector_results else 0)+(1-vector_weight if lexical_results else 0)
    for hit in hits.values():
        # RRF 的理论首位上限归一化到 [0,1]，与 cosine redundancy 使用相近尺度。
        hit.relevance = hit.rrf_score*(rrf_k+1)/active_weight
        # 纯向量对照保留原有 cosine relevance / MMR 行为。
        if vector_weight == 1:
            hit.relevance = hit.vector_score
    candidates = sorted(hits, key=lambda i: (-hits[i].rrf_score, i))[:top_k]
    return [hits[i] for i in candidates]


def merge_rounds(*rounds):
    """按 chunk ID 合并，保留最强召回分数和全部命中轮次，不截断。"""
    merged = {}
    for name, hits in rounds:
        for hit in hits:
            key = hit.chunk.chunk_id
            if key not in merged:
                merged[key] = replace(hit, retrieval_rounds=[])
            elif hit.relevance > merged[key].relevance:
                merged[key] = replace(hit, retrieval_rounds=list(merged[key].retrieval_rounds))
            if name not in merged[key].retrieval_rounds:
                merged[key].retrieval_rounds.append(name)
    # 首次出现顺序用于后续重排同分时稳定排序。各轮完整分数由调用方分别保存。
    return list(merged.values())


def select_mmr(snapshot, candidates, mmr_k=4, mmr_lambda=0.7):
    """在 Cross-Encoder 之前压缩候选，不修改输入。

    relevance 来自最强命中轮次的召回分数，裁剪到 [0,1]。
    redundancy 使用快照内归一化向量的余弦相似度，负值按 0 处理。
    第一项始终选最相关者；后续按 MMR 选择，同分优先相关性及输入顺序。
    """
    if mmr_k < 1 or not 0 <= mmr_lambda <= 1:
        raise ValueError('mmr_k 至少为 1；MMR lambda 必须在 [0,1]')
    if not candidates:
        return []
    scores = np.asarray([hit.relevance for hit in candidates], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('召回相关性必须为有限值')
    relevance = np.clip(scores, 0.0, 1.0)
    vector_ids = {chunk.chunk_id: i for i, chunk in enumerate(snapshot.chunks)}
    vectors = np.asarray([snapshot.vectors[vector_ids[hit.chunk.chunk_id]] for hit in candidates])
    similarities = np.clip(vectors @ vectors.T, 0.0, 1.0)
    chosen, remaining, selected = [], list(range(len(candidates))), []
    while remaining and len(selected) < mmr_k:
        def metrics(i):
            redundancy = float(similarities[i, chosen].max()) if chosen else 0.0
            score = mmr_lambda*float(relevance[i])-(1-mmr_lambda)*redundancy
            return score, redundancy
        best = max(remaining, key=lambda i: (
            metrics(i)[0] if chosen else float(relevance[i]), float(relevance[i]), -i))
        score, redundancy = metrics(best)
        selected.append(replace(candidates[best], mmr_relevance=float(relevance[best]),
                                redundancy=redundancy, mmr_score=score))
        chosen.append(best)
        remaining.remove(best)
    return selected


def retrieve(snapshot, query_vector, fetch_k=20, top_k=4, mmr_lambda=0.7, *,
             query=None, vector_weight=0.5, rrf_k=60):
    """旧版单轮 RRF + MMR，仅用于对照测试与评估。"""
    if not 0 <= mmr_lambda <= 1 or not 1 <= top_k <= fetch_k:
        raise ValueError('MMR lambda 必须在 [0,1]；要求 1 <= top_k <= fetch_k')
    recalled = recall(snapshot, query_vector, fetch_k, fetch_k, query=query,
                      vector_weight=vector_weight, rrf_k=rrf_k)
    hits = {i: hit for i, hit in enumerate(recalled)}
    # MMR 需要快照内的向量行号，而不是召回排名。
    vector_ids = {chunk.chunk_id: i for i, chunk in enumerate(snapshot.chunks)}
    candidates = list(hits)
    selected, remaining, chosen_ids = [], list(candidates), []
    while remaining and len(selected) < top_k:
        ranked = []
        for i in remaining:
            relevance = hits[i].relevance
            vector_i = vector_ids[hits[i].chunk.chunk_id]
            redundancy = max(float(snapshot.vectors[vector_i] @ snapshot.vectors[vector_ids[hits[j].chunk.chunk_id]]) for j in chosen_ids) if chosen_ids else 0.0
            score = mmr_lambda*relevance-(1-mmr_lambda)*redundancy
            ranked.append((score,relevance,-vector_i,i,redundancy))
        # 第一项选最相关者；后续项才考虑多样性，避免 lambda=0 首项无意义。
        best = max(ranked, key=lambda r:(r[1],r[2])) if not chosen_ids else max(ranked)
        score,_,_,i,redundancy = best
        selected.append(replace(hits[i], redundancy=redundancy, mmr_score=score))
        chosen_ids.append(i)
        remaining.remove(i)
    return recalled, selected
