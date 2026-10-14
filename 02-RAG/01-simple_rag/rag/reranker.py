"""本地 Cross-Encoder 对原始问题与候选文本联合打分。"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


class Reranker:
    def __init__(self, model_path: Path, device='auto', batch_size=4, max_length=512):
        from sentence_transformers import CrossEncoder
        if not model_path.is_dir():
            raise FileNotFoundError(f'重排模型不存在：{model_path}；请先运行 scripts/download_reranker.py')
        if batch_size < 1 or max_length < 32:
            raise ValueError('reranker batch size 至少为 1，max length 至少为 32')
        torch.set_num_threads(1)
        self.device = ('mps' if torch.backends.mps.is_available() else 'cpu') if device == 'auto' else device
        self.batch_size = batch_size
        self.model = CrossEncoder(str(model_path), device=self.device, local_files_only=True,
                                  max_length=max_length, activation_fn=torch.nn.Identity(),
                                  model_kwargs={'torch_dtype': torch.float32})
        if self.model.num_labels != 1:
            raise ValueError('重排模型必须为每个 query/document 对输出一个相关性分数')

    def rerank(self, query, hits, top_n=2):
        if not query.strip() or top_n < 1:
            raise ValueError('重排 query 不能为空；top_n 至少为 1')
        if not hits:
            return [], []
        # 保留标题和本组问题，避免脱离问答语境；只在此处进行联合编码。
        pairs = [(query, hit.chunk.embedding_text) for hit in hits]
        scores = np.asarray(self.model.predict(pairs, batch_size=self.batch_size,
                                               show_progress_bar=False, convert_to_numpy=True))
        if scores.shape != (len(hits),) or not np.isfinite(scores).all():
            raise ValueError(f'Cross-Encoder 返回无效分数：形状 {scores.shape}')
        ranked = [replace(hit, cross_encoder_score=float(score)) for hit, score in zip(hits, scores)]
        ranked.sort(key=lambda hit: -hit.cross_encoder_score)
        return ranked, ranked[:top_n]
