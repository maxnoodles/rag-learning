"""保持每组问答的边界；长问答切片时重复问题作为 embedding 上下文。"""
import hashlib
import re

from .schemas import Chunk, Document

CHUNKER_VERSION = 'qa-character-spans-v1'


def embedding_input(document: Document, body: str) -> str:
    question = f'本组问题：{document.question}\n' if document.question else ''
    # 已经手工构造文档前缀，调用 encode 时不要再指定 Document prompt。
    return f'title: {document.title} | text: {question}{body}'


def make_chunks(documents: list[Document], tokenizer, max_tokens=512, overlap_tokens=64):
    if max_tokens < 32 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError('chunk tokens 至少 32；overlap 必须在 [0, chunk tokens) 内')
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True))
    chunks = []
    for doc in documents:
        if count(embedding_input(doc, '')) >= max_tokens-8:
            raise ValueError(f'{doc.source} 的标题或问题太长，无法在 token 上限内保留问题')
        start, ordinal = 0, 0
        while start < len(doc.text):
            lo, hi = start+1, len(doc.text)
            # 对原文字符范围二分，而不是 decode token 后丢失原文定位。
            best = start
            while lo <= hi:
                mid = (lo+hi)//2
                if count(embedding_input(doc, doc.text[start:mid])) <= max_tokens:
                    best, lo = mid, mid+1
                else:
                    hi = mid-1
            if best <= start:
                raise ValueError(f'{doc.source} 无法切分；检查 token 上限')
            end = best
            # 优先在段落或句子边界断开，但避免产生极短片段。
            if end < len(doc.text):
                boundaries = [m.end()+start for m in re.finditer(r'\n\n|[。！？]\s*', doc.text[start:end])]
                boundaries = [b for b in boundaries if b >= start+(end-start)//2]
                if boundaries:
                    end = boundaries[-1]
            body = doc.text[start:end]
            value = embedding_input(doc, body)
            digest = hashlib.sha256(f'{doc.doc_id}:{start}:{end}:{body}'.encode()).hexdigest()[:20]
            chunks.append(Chunk(digest, doc.doc_id, ordinal, start, end, body, value, count(value)))
            ordinal += 1
            if end == len(doc.text):
                break
            # 用实际 tokenizer 测 overlap 的原文后缀；start 至少向前走一个字符。
            lo, hi, next_start = start+1, end, end
            while lo <= hi:
                mid = (lo+hi)//2
                if count(doc.text[mid:end]) <= overlap_tokens:
                    next_start, hi = mid, mid-1
                else:
                    lo = mid+1
            start = max(start+1, next_start)
    return chunks
