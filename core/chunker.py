from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_id: int
    page_number: int
    element_type: str  # "text", "table", "code"
    headings: list[str] = field(default_factory=list)
    context_text: str = ""
