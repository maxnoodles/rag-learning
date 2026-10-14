"""FAISS + SQLite 不可变快照；完整构建后才原子切换 CURRENT。"""
import hashlib
import importlib.metadata
import json
import os
import sqlite3
import uuid
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .chunker import CHUNKER_VERSION
from .document_parser import PARSER_VERSION, source_files
from .lexical_index import BM25Index, LEXICAL_VERSION
from .schemas import Chunk, Document

STORE_VERSION = 'faiss-sqlite-v1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(doc_directory, model_directory, chunk_tokens, overlap_tokens):
    model_files = sorted(p for p in model_directory.rglob('*') if p.is_file()
                         and p.suffix in ('.json', '.safetensors', '.model', '.jinja'))
    if not model_files or not (model_directory/'model.safetensors').exists():
        raise ValueError(f'模型目录缺少权重或配置：{model_directory}')
    config = dict(store_version=STORE_VERSION, parser=PARSER_VERSION, chunker=CHUNKER_VERSION,
                  lexical=LEXICAL_VERSION,
                  chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
                  model_files={str(p.relative_to(model_directory)):sha256(p) for p in model_files},
                  documents={str(p.relative_to(doc_directory)):sha256(p) for p in source_files(doc_directory)},
                  dimension=768, dtype='float32', normalize=True, metric='inner_product',
                  document_prompt='title: {title} | text: {question_and_body}', query_prompt='SearchQuery',
                  dependencies={name:importlib.metadata.version(name) for name in
                                ('sentence-transformers', 'transformers', 'torch', 'faiss-cpu',
                                 'langchain-text-splitters', 'jieba')})
    value = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return value, config


class Snapshot:
    def __init__(self, generation, index, documents, chunks, manifest):
        self.generation = generation
        self.index = index
        self.documents = {d.doc_id:d for d in documents}
        self.chunks = chunks
        self.manifest = manifest
        self.vectors = index.reconstruct_n(0, index.ntotal)
        # 由 SQLite 恢复的同一批 chunks 构建，ID 与 FAISS 完全一致，不重复做 embedding。
        self.bm25 = BM25Index(f'{self.documents[c.doc_id].title}\n'
                              f'{self.documents[c.doc_id].question}\n{c.text}' for c in chunks)


def load_current(root: Path, expected_fingerprint):
    pointer = root/'CURRENT'
    if not pointer.exists():
        return None, '尚无索引缓存'
    try:
        generation = pointer.read_text().strip()
        if not generation or Path(generation).name != generation or generation in ('.', '..'):
            raise ValueError('CURRENT 非法')
        folder = root/'generations'/generation
        manifest = json.loads((folder/'manifest.json').read_text())
        if manifest['fingerprint'] != expected_fingerprint:
            return None, '文档、模型、依赖版本或切分配置发生变化'
        for filename, digest in manifest['checksums'].items():
            if filename not in ('vectors.faiss', 'metadata.sqlite') or sha256(folder/filename) != digest:
                raise ValueError(f'缓存校验失败：{filename}')
        if set(manifest['checksums']) != {'vectors.faiss', 'metadata.sqlite'}:
            raise ValueError('缓存文件不完整')
        # 只有本程序写入且校验通过的本地文件才交给 FAISS 读取。
        index = faiss.read_index(str(folder/'vectors.faiss'))
        with sqlite3.connect(f'file:{folder / "metadata.sqlite"}?mode=ro', uri=True) as db:
            documents = [Document(**json.loads(r[0])) for r in db.execute('SELECT payload FROM documents ORDER BY rowid')]
            chunks = [Chunk(**json.loads(r[0])) for r in db.execute('SELECT payload FROM chunks ORDER BY vector_id')]
        if index.d != 768 or index.ntotal != len(chunks) or len(chunks) != manifest['chunk_count']:
            raise ValueError('索引维度或 ID 映射数量不一致')
        if len(documents) != manifest['document_count']:
            raise ValueError('文档数量不一致')
        docs = {d.doc_id:d for d in documents}
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError('chunk ID 重复')
        for c in chunks:
            if c.doc_id not in docs or docs[c.doc_id].text[c.start:c.end] != c.text:
                raise ValueError('chunk 原文范围不一致')
        snapshot = Snapshot(generation, index, documents, chunks, manifest)
        if not np.isfinite(snapshot.vectors).all() or not np.allclose(np.linalg.norm(snapshot.vectors, axis=1), 1, atol=1e-4):
            raise ValueError('缓存向量不合法')
        return snapshot, '缓存指纹与文件校验均通过，跳过解析、切分与文档 embedding'
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, sqlite3.Error) as exc:
        return None, f'缓存损坏或不完整：{exc}'


def save_snapshot(root, identity, config, documents, chunks, vectors):
    if not chunks or vectors.shape != (len(chunks), 768) or not np.isfinite(vectors).all():
        raise ValueError('无法持久化无效向量')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('向量未归一化')
    generation = uuid.uuid4().hex
    folder = root/'generations'/generation
    folder.mkdir(parents=True)
    index = faiss.IndexFlatIP(768)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    faiss.write_index(index, str(folder/'vectors.faiss'))
    with sqlite3.connect(folder/'metadata.sqlite') as db:
        db.execute('CREATE TABLE documents (doc_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        db.execute('CREATE TABLE chunks (vector_id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, payload TEXT NOT NULL)')
        db.executemany('INSERT INTO documents VALUES (?, ?)',
                       [(d.doc_id,json.dumps(asdict(d),ensure_ascii=False)) for d in documents])
        db.executemany('INSERT INTO chunks VALUES (?, ?, ?)',
                       [(i,c.chunk_id,json.dumps(asdict(c),ensure_ascii=False)) for i,c in enumerate(chunks)])
    # sqlite connect 上下文只提交；显式关闭连接之后文件成为只读快照。
    db.close()
    scores, ids = index.search(vectors[:1], 1)
    if ids[0,0] < 0 or scores[0,0] < 0.999:
        raise ValueError('索引自检失败')
    manifest = dict(fingerprint=identity, config=config, document_count=len(documents),
                    chunk_count=len(chunks), checksums={n:sha256(folder/n) for n in
                                                     ('vectors.faiss', 'metadata.sqlite')})
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    for filename in ('vectors.faiss','metadata.sqlite','manifest.json'):
        with (folder/filename).open('rb') as file:
            os.fsync(file.fileno())
    temp = root/f'.CURRENT-{generation}'
    with temp.open('w') as file:
        file.write(generation+'\n'); file.flush(); os.fsync(file.fileno())
    os.replace(temp, root/'CURRENT')
    return Snapshot(generation, index, documents, chunks, manifest)
