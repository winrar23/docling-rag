"""Генераторы мини-PDF для герметичных тестов детекта текстового слоя (без Docling).

make_text_pdf — рукописный минимальный PDF с настоящим текстовым слоем
(Helvetica, latin-1); make_empty_pdf — страницы без контента (эмуляция скана).
Оба проверены на читаемость pypdfium2.
"""
import io


def make_text_pdf(text: str, pages: int = 1) -> bytes:
    """Валидный PDF: на каждой из pages страниц строка text (латиница) текстовым слоем."""
    objs: list[bytes] = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages)).encode()
    objs.append(b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(pages).encode() + b" >>")
    font_num = 3 + 2 * pages
    for i in range(pages):
        content_num = 3 + pages + i
        objs.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842]"
            b" /Resources << /Font << /F1 " + str(font_num).encode() + b" 0 R >> >>"
            b" /Contents " + str(content_num).encode() + b" 0 R >>"
        )
    for i in range(pages):
        stream = b"BT /F1 12 Tf 50 700 Td (" + text.encode("latin-1") + b") Tj ET"
        objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += str(n).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (b"trailer\n<< /Size " + str(len(objs) + 1).encode() + b" /Root 1 0 R >>\n"
            b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF\n")
    return bytes(out)


def make_empty_pdf(pages: int = 1) -> bytes:
    """PDF из pages пустых страниц — текстового слоя нет (эмуляция скана)."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(595, 842)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
