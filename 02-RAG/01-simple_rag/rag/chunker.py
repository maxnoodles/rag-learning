"""LangChain 递归多级切分；保留问答边界、原文范围和 embedding token 上限。"""
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .schemas import Chunk, Document

CHUNKER_VERSION = 'qa-langchain-recursive-spans-v2'
# 从粗到细；中英文句子、子句都保留分隔符，最后才退化到字符。
SEPARATORS = ['\n\n', '\n', '。', '！', '？', '. ', '! ', '? ', '；', '; ',
              '，', ', ', '、', ' ', '\u200b', '']


def embedding_input(document: Document, body: str) -> str:
    question = f'本组问题：{document.question}\n' if document.question else ''
    # 已经手工构造文档前缀，调用 encode 时不要再指定 Document prompt。
    return f'title: {document.title} | text: {question}{body}'


def make_chunks(documents: list[Document], tokenizer, max_tokens=512, overlap_tokens=64):
    if max_tokens < 32 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError('chunk tokens 至少 32；overlap 必须在 [0, chunk tokens) 内')
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True))
    body_count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    chunks = []
    for doc in documents:
        prefix_tokens = count(embedding_input(doc, ''))
        if prefix_tokens >= max_tokens-8:
            raise ValueError(f'{doc.source} 的标题或问题太长，无法在 token 上限内保留问题')
        # 预留前缀、overlap 和少量 BPE 拼接余量，最后仍检查完整 embedding 输入。
        body_budget = max_tokens-prefix_tokens-8
        overlap = min(overlap_tokens, body_budget-1)
        core_budget = body_budget-overlap
        while True:
            splitter = RecursiveCharacterTextSplitter(
                separators=SEPARATORS, chunk_size=core_budget, chunk_overlap=0,
                length_function=body_count, keep_separator='end', strip_whitespace=False,
                is_separator_regex=False,
            )
            bodies = splitter.split_text(doc.text)
            if all(count(embedding_input(doc, body)) <= max_tokens for body in bodies):
                break
            if core_budget == 1:
                raise ValueError(f'{doc.source} 无法切分；检查 token 上限')
            core_budget = max(1, core_budget//2)
        # 核心块不重叠，因而可精确累计字符位置；再用原文后缀加入 token overlap。
        # LangChain 的 add_start_index 按字符减 chunk_overlap，不能直接用于 token 单位。
        cursor = 0
        for ordinal, core in enumerate(bodies):
            if doc.text[cursor:cursor+len(core)] != core:
                raise ValueError(f'{doc.source} 递归切分丢失原文定位')
            end = cursor+len(core)
            start = cursor
            # overlap 为目标上限；过长问题或完整 embedding 预算不足时缩短。
            previous_start = chunks[-1].start if ordinal else cursor
            for candidate in range(cursor-1, previous_start-1, -1):
                if body_count(doc.text[candidate:cursor]) > overlap:
                    break
                if count(embedding_input(doc, doc.text[candidate:end])) > max_tokens:
                    break
                start = candidate
            body = doc.text[start:end]
            value = embedding_input(doc, body)
            digest = hashlib.sha256(f'{doc.doc_id}:{start}:{end}:{body}'.encode()).hexdigest()[:20]
            chunks.append(Chunk(digest, doc.doc_id, ordinal, start, end, body, value, count(value)))
            cursor = end
        if cursor != len(doc.text):
            raise ValueError(f'{doc.source} 递归切分未覆盖全文')
    return chunks
