from dataclasses import dataclass, field

from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument


@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_id: int
    page_number: int
    element_type: str  # "text", "table", "code"
    headings: list[str] = field(default_factory=list)
    context_text: str = ""


def _extract_element_type(doc_chunk) -> str:
    """Map DocChunk's first doc_item label to our element_type string."""
    try:
        label = doc_chunk.meta.doc_items[0].label.value
    except (IndexError, AttributeError):
        return "text"
    if label == "table":
        return "table"
    if label == "code":
        return "code"
    return "text"


def _extract_page_number(doc_chunk) -> int:
    """Extract page number from first doc_item's provenance."""
    try:
        prov = doc_chunk.meta.doc_items[0].prov
        if prov and len(prov) > 0:
            return int(prov[0].page_no)
    except (IndexError, AttributeError, TypeError, ValueError):
        pass
    return 1


def chunk_document(
    dl_doc: DoclingDocument,
    source_file: str,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> list[Chunk]:
    """
    Chunk a DoclingDocument using Docling's HybridChunker.

    Returns list of Chunk objects with heading context for embedding.
    HybridChunker splits by document structure, respects token limits
    of the embedding model, and merges small peer chunks.
    """
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        f"sentence-transformers/{embedding_model}"
    )
    chunker = HybridChunker(tokenizer=tokenizer)

    chunks: list[Chunk] = []
    for chunk_id, doc_chunk in enumerate(chunker.chunk(dl_doc)):
        context_text = chunker.contextualize(doc_chunk)
        headings = list(doc_chunk.meta.headings or [])

        chunks.append(Chunk(
            text=doc_chunk.text,
            source_file=source_file,
            chunk_id=chunk_id,
            page_number=_extract_page_number(doc_chunk),
            element_type=_extract_element_type(doc_chunk),
            headings=headings,
            context_text=context_text,
        ))

    return chunks
