import sys
from dataclasses import dataclass, field
from functools import lru_cache

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


@lru_cache(maxsize=4)
def _get_chunker(embedding_model: str, max_tokens: int) -> HybridChunker:
    """
    Get or create a cached HybridChunker for the given (model, max_tokens) pair.

    Cached per (model, max_tokens) to avoid repeated tokenizer initialization.
    Имя с org ("deepvk/USER-bge-m3") используется как есть; короткое имя получает
    префикс sentence-transformers/. Явный max_tokens обязателен: у bge-m3 окно
    8192 — авто-лимит дал бы чанки, убивающие гранулярность поиска.
    """
    model_id = embedding_model if "/" in embedding_model else f"sentence-transformers/{embedding_model}"
    # model_max_length=sys.maxsize: токенайзер здесь только СЧИТАЕТ токены (лимит чанков —
    # max_tokens), но transformers сравнивает длину с model_max_length из конфига модели и
    # шумит "Token indices sequence length is longer than ..." на секциях длиннее окна.
    tokenizer = HuggingFaceTokenizer.from_pretrained(model_id, max_tokens=max_tokens, model_max_length=sys.maxsize)
    return HybridChunker(tokenizer=tokenizer)


def chunk_document(
    dl_doc: DoclingDocument,
    source_file: str,
    embedding_model: str = "deepvk/USER-bge-m3",
    max_tokens: int = 512,
) -> list[Chunk]:
    """
    Chunk a DoclingDocument using Docling's HybridChunker.

    Returns list of Chunk objects with heading context for embedding.
    HybridChunker splits by document structure, respects token limits
    of the embedding model, and merges small peer chunks.
    """
    chunker = _get_chunker(embedding_model, max_tokens)

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
