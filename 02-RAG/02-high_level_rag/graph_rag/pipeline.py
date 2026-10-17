"""递归切块 → LLMGraphTransformer → 保留来源的 GraphDocument。"""
import hashlib
import json
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_neo4j.graphs.graph_document import GraphDocument
from langchain_neo4j.graph_transformers.llm import LLMGraphTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .codex_model import CodexChatModel

NODE_TYPES = ['Person', 'Company', 'Department', 'Project', 'Technology']
RELATIONSHIPS = [
    ('Department', 'PART_OF', 'Company'),
    ('Person', 'WORKS_IN', 'Department'),
    ('Person', 'LEADS', 'Project'),
    ('Person', 'PARTICIPATES_IN', 'Project'),
    ('Person', 'KNOWS', 'Technology'),
    ('Project', 'OWNED_BY', 'Company'),
    ('Project', 'MAINTAINED_BY', 'Department'),
    ('Project', 'USES', 'Technology'),
    ('Project', 'DEPENDS_ON', 'Project'),
]
ENCODING = 'o200k_base'


def load_documents(input_dir):
    input_dir = Path(input_dir)
    documents = []
    for path in sorted(input_dir.rglob('*.txt')):
        text = path.read_text(encoding='utf-8').strip()
        if text:
            relative = path.relative_to(input_dir).as_posix()
            documents.append(Document(page_content=text, metadata={
                'document_id': relative, 'source': relative,
            }))
    if not documents:
        raise ValueError(f'未找到非空的 .txt 文档：{input_dir}')
    return documents


def split_documents(documents, chunk_size=60, chunk_overlap=10):
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError('要求 0 <= chunk_overlap < chunk_size')
    # 递归按段落、换行、中文句末、空格和字符切分，大小以 token 衡量。
    # 这不是依次对同一段文本进行多轮 tokenizer 分词。
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', '。', '！', '？', '；', ' ', ''],
        keep_separator='end',
    )
    encoder = tiktoken.get_encoding(ENCODING)
    chunks = []
    for document in documents:
        for index, chunk in enumerate(splitter.split_documents([document])):
            digest = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:12]
            chunk_id = f'{document.metadata["document_id"]}:{index:03d}:{digest}'
            chunk.id = chunk_id
            chunk.metadata.update(chunk_id=chunk_id, chunk_index=index,
                                  token_count=len(encoder.encode(chunk.page_content)),
                                  tokenizer=ENCODING)
            chunks.append(chunk)
    return chunks


def create_transformer(model=None, timeout=180, llm=None):
    return LLMGraphTransformer(
        llm=llm if llm is not None else CodexChatModel(model=model, timeout=timeout),
        allowed_nodes=NODE_TYPES, allowed_relationships=RELATIONSHIPS,
        strict_mode=True, relationship_properties=['evidence'],
        additional_instructions=(
            '只抽取当前文本明确陈述的实体和直接关系，不使用外部知识、不补全跨片段关系。'
            '实体名称保持原文写法；实体类型使用给定的英文标签。'
            '每条关系都必须有 evidence 属性，其值为包含两端实体及该关系的原文连续引文。'
            '关系方向严格按 schema：人→部门/项目/技术，项目→公司/维护部门/技术/依赖项目，部门→公司。'
            '负责=LEADS，参与=PARTICIPATES_IN，熟悉=KNOWS；使用与熟悉不要混淆。'
            '不把用途、编程语言、缓存技术、容器技术等类别说明当成额外实体或关系。'
            '没有关系时仍保留能识别的实体，返回空的关系列表。'
        ),
    )


def validate_graph_document(graph):
    """防止空结果、悬空边以及没有原文引文的关系悄悄落盘。"""
    text = graph.source.page_content
    nodes = {(node.id, node.type) for node in graph.nodes}
    if not nodes:
        raise ValueError(f'未抽取到节点：{graph.source.metadata["chunk_id"]}')
    for node in graph.nodes:
        if not node.id or node.id.casefold() not in text.casefold():
            raise ValueError(f'节点未出现在当前原文中：{node.id}')
    for rel in graph.relationships:
        if (rel.source.id, rel.source.type) not in nodes or (rel.target.id, rel.target.type) not in nodes:
            raise ValueError('关系端点未包含在节点列表中')
        if (rel.source.type, rel.type, rel.target.type) not in RELATIONSHIPS:
            raise ValueError(f'关系不符合 schema：{rel.type}')
        evidence = rel.properties.get('evidence')
        if not isinstance(evidence, str) or not evidence.strip() or evidence not in text:
            raise ValueError(f'关系缺少有效原文引文：{rel.type}')
        if any(node.id.casefold() not in evidence.casefold() for node in (rel.source, rel.target)):
            raise ValueError(f'引文未包含关系两端的实体：{rel.type}')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def build_graph_documents(chunks, transformer, output_dir):
    graphs = []
    for index, chunk in enumerate(chunks, 1):
        print(f'[{index}/{len(chunks)}] 抽取 {chunk.metadata["chunk_id"]}', flush=True)
        # 返回值已经是 List[GraphDocument]，无需再包一层同名对象。
        graph = transformer.convert_to_graph_documents([chunk])[0]
        validate_graph_document(graph)
        graph.nodes.sort(key=lambda node: (node.type, node.id))
        graph.relationships.sort(key=lambda rel: (rel.source.id, rel.type, rel.target.id))
        graphs.append(graph)
    save_json(Path(output_dir) / 'graph_documents.json',
              [graph.model_dump(mode='json') for graph in graphs])
    return graphs


def summarize(graphs):
    nodes = {(node.type, node.id) for graph in graphs for node in graph.nodes}
    edges = {(rel.source.type, rel.source.id, rel.type, rel.target.type, rel.target.id)
             for graph in graphs for rel in graph.relationships}
    return {'graph_documents': len(graphs), 'unique_nodes': len(nodes),
            'unique_relationships': len(edges)}


def load_graph_documents(path):
    return [GraphDocument.model_validate(item)
            for item in json.loads(Path(path).read_text(encoding='utf-8'))]
