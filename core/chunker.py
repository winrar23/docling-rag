from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_id: int
    page_number: int
    element_type: str  # "text", "table", "code"


def chunk_elements(
    elements: list[dict[str, Any]],
    source_file: str,
    chunk_size: int = 3200,   # characters (≈800 tokens × 4 chars/token)
    overlap: int = 320,        # characters (≈80 tokens)
) -> list[Chunk]:
    """
    Splits list of Docling elements into Chunk objects.
    Tables and code blocks → atomic chunks.
    Text elements → accumulated to chunk_size chars with overlap.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    chunks: list[Chunk] = []
    chunk_id = 0
    text_buffer = ""
    buffer_page = 1

    def flush_buffer() -> None:
        nonlocal chunk_id, text_buffer, buffer_page
        if text_buffer.strip():
            chunks.append(Chunk(
                text=text_buffer.strip(),
                source_file=source_file,
                chunk_id=chunk_id,
                page_number=buffer_page,
                element_type="text",
            ))
            chunk_id += 1
        text_buffer = ""

    for element in elements:
        etype = element.get("type", "text")
        text = element.get("text", "")
        page = element.get("page", 1)

        if etype in ("table", "code"):
            flush_buffer()
            if text.strip():
                chunks.append(Chunk(
                    text=text.strip(),
                    source_file=source_file,
                    chunk_id=chunk_id,
                    page_number=page,
                    element_type=etype,
                ))
                chunk_id += 1
        else:
            if not text_buffer:
                buffer_page = page
            text_buffer += text + " "

            while len(text_buffer) > chunk_size:
                cut = text_buffer.rfind(". ", 0, chunk_size)
                if cut == -1 or cut + 2 <= overlap:   # boundary too close — fall back to hard cut
                    cut = chunk_size
                else:
                    cut += 2

                chunk_text = text_buffer[:cut].strip()
                carry = text_buffer[max(0, cut - overlap):cut]
                chunks.append(Chunk(
                    text=chunk_text,
                    source_file=source_file,
                    chunk_id=chunk_id,
                    page_number=buffer_page,
                    element_type="text",
                ))
                chunk_id += 1
                text_buffer = carry + text_buffer[cut:]
                buffer_page = page

    flush_buffer()
    return chunks
