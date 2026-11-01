from dataclasses import asdict, dataclass, field


@dataclass
class Section:
    section_id: str
    heading: str
    order: int
    kind: str
    text: str


@dataclass
class Document:
    document_id: str
    relative_path: str
    title: str
    category: str
    api_path: str
    method: str
    source_url: str
    content_hash: str
    content: str
    sections: list[Section]
    is_legacy: bool = False
    verified_update: bool = False
    source_revision: str = ""
    updated_at: str = ""
    synced_at: str = ""
    update_basis: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["sections"] = [Section(**s) for s in value["sections"]]
        return cls(**value)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    parent_id: str
    section_order: int
    order: int
    title: str
    heading: str
    category: str
    api_path: str
    method: str
    kind: str
    text: str
    embedding_text: str
    token_count: int

    def to_dict(self):
        return asdict(self)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float = 0.0
    vector_similarity: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    channels: list[str] = field(default_factory=list)


@dataclass
class ContextBundle:
    text: str
    sources: list[dict]
    estimated_tokens: int
    notices: list[str] = field(default_factory=list)
