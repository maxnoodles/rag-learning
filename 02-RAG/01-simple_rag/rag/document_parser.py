"""TXT / Markdown parser 注册表，偏移量始终对应规范化正文。"""
import hashlib
import json
import re
from pathlib import Path

from .schemas import Document

PARSER_VERSION = 'txt-md-v1'


def parse_text(path: Path, relative: str) -> Document:
    raw = path.read_text(encoding='utf-8-sig')  # 明确报编码错误，不静默替换字符。
    text = raw.replace('\r\n', '\n').replace('\r', '\n').strip()
    metadata = {}
    # 聊天样例使用 JSON frontmatter；普通 Markdown frontmatter 不要求此格式。
    match = re.match(r'\A---\n(\{[^\n]*\})\n---\n?', text)
    if match:
        metadata = json.loads(match.group(1))
        text = text[match.end():].strip()
    if not text:
        raise ValueError(f'空文档不能成功入库：{relative}')
    question = ''
    qa = re.search(r'^## 用户\n\n(.*?)\n\n## 助手\n', text, re.M | re.S)
    if qa:
        question = qa.group(1).strip()
    title = metadata.get('title') or path.stem
    doc_id = hashlib.sha256(relative.encode()).hexdigest()[:16]
    return Document(doc_id, relative, title, text, question, metadata)


PARSERS = {'.txt': parse_text, '.md': parse_text, '.markdown': parse_text}


def source_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f'文档目录不存在：{directory}')
    files = sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in PARSERS)
    if not files:
        raise ValueError(f'没有 TXT / Markdown 文档：{directory}')
    return files


def parse_documents(directory: Path) -> list[Document]:
    return [PARSERS[p.suffix.lower()](p, p.relative_to(directory).as_posix())
            for p in source_files(directory)]
