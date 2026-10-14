"""同一快照的 BM25 倒排索引；jieba 中文分词，英文/代码标识符统一小写。"""
import logging
import math
import re
from collections import Counter, defaultdict

import jieba
import numpy as np

LEXICAL_VERSION = 'jieba-search-bm25-positive-idf-v1'
jieba.setLogLevel(logging.WARNING)


def tokenize(text):
    return [term for term in jieba.cut_for_search(text.casefold(), HMM=False)
            if re.search(r'\w', term)]


class BM25Index:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)
        lengths = []
        for i, text in enumerate(texts):
            terms = tokenize(text)
            lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                self.postings[term].append((i, frequency))
        self.lengths = np.asarray(lengths, dtype=np.float32)
        self.size = len(lengths)
        self.avg_length = max(float(self.lengths.mean()), 1.0) if self.size else 1.0

    def search(self, query, k):
        scores = np.zeros(self.size, dtype=np.float64)
        for term in set(tokenize(query)):
            postings = self.postings.get(term, ())
            if not postings:
                continue
            # log(1 + ...) 形式的 BM25 IDF，常见词也有正分，空匹配保持 0。
            df = len(postings)
            idf = math.log1p((self.size-df+0.5)/(df+0.5))
            for i, tf in postings:
                norm = self.k1*(1-self.b+self.b*self.lengths[i]/self.avg_length)
                scores[i] += idf*tf*(self.k1+1)/(tf+norm)
        candidates = np.flatnonzero(scores > 0)
        ranked = sorted(candidates, key=lambda i: (-scores[i], int(i)))[:k]
        return [(int(i), float(scores[i])) for i in ranked]
