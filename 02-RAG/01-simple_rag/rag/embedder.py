"""单例模型持有者，完全本地推理，查询与文档使用不同任务前缀。"""
import os
from pathlib import Path

# 必须在 PyTorch / FAISS 导入前设置，规避本机既有 OpenMP 冲突。
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'

import faiss
import numpy as np
import torch


class Embedder:
    dimension = 768

    def __init__(self, model_path: Path, device='auto'):
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)
        self.device = ('mps' if torch.backends.mps.is_available() else 'cpu') if device == 'auto' else device
        self.model = SentenceTransformer(str(model_path), device=self.device, local_files_only=True,
                                        config_kwargs={'audio_config': None, 'vision_config': None},
                                        model_kwargs={'torch_dtype': torch.float32})
        self.tokenizer = self.model.tokenizer

    def _encode(self, texts, **kwargs):
        vectors = np.asarray(self.model.encode(texts, batch_size=8, normalize_embeddings=True,
                                              convert_to_numpy=True, show_progress_bar=False,
                                              **kwargs), dtype=np.float32)
        if vectors.shape != (len(texts), self.dimension):
            raise ValueError(f'模型向量形状错误：{vectors.shape}，应为 {(len(texts), self.dimension)}')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or np.any(norms == 0):
            raise ValueError('模型产生 NaN / Inf 或零向量')
        return np.ascontiguousarray(vectors/norms, dtype=np.float32)

    def documents(self, chunks):
        return self._encode([c.embedding_text for c in chunks])

    def query(self, query):
        prefix = self.model.prompts['SearchQuery']
        length = len(self.tokenizer.encode(prefix+query, add_special_tokens=True))
        if length > 4096:
            raise ValueError('query 超过学习版 4096 tokens 上限；请缩短问题')
        return self._encode([query], prompt_name='SearchQuery')[0]
