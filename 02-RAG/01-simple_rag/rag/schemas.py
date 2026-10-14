from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    text: str
    question: str = ''
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    start: int
    end: int
    text: str
    embedding_text: str
    token_count: int


@dataclass
class Hit:
    chunk: Chunk
    relevance: float
    redundancy: float = 0.0
    mmr_score: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    retrieval_rounds: list[str] = field(default_factory=list)
    cross_encoder_score: float | None = None


@dataclass
class Evidence:
    citation_id: str
    text: str
    sources: list[dict]
    chunk_ids: list[str]
    expanded: bool
    start: int
    end: int
