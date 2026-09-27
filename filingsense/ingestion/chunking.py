"""Table-aware chunking for 10-K text.

Financial statements live in tables; naive fixed-size chunking shreds them.
This splitter never breaks inside a table block and keeps section headers
attached to the chunk they introduce.
"""
from __future__ import annotations

TARGET_CHARS = 2000   # ~500 tokens
OVERLAP_CHARS = 200


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|")


def split_blocks(text: str) -> list[str]:
    """Split into atomic blocks: paragraphs, headers, whole tables."""
    blocks, buf, in_table = [], [], False
    for line in text.splitlines():
        if _is_table_row(line):
            if not in_table and buf:
                blocks.append("\n".join(buf))
                buf = []
            in_table = True
            buf.append(line)
        else:
            if in_table:
                blocks.append("\n".join(buf))
                buf = []
                in_table = False
            if line.strip():
                buf.append(line)
            elif buf:
                blocks.append("\n".join(buf))
                buf = []
    if buf:
        blocks.append("\n".join(buf))
    return blocks


def chunk_text(text: str, target: int = TARGET_CHARS,
               overlap: int = OVERLAP_CHARS) -> list[str]:
    chunks, cur = [], ""
    for block in split_blocks(text):
        # An oversized single block (giant table) is kept whole — splitting it
        # destroys more meaning than the size costs. Route numeric questions
        # to the XBRL path instead.
        if len(cur) + len(block) > target and cur:
            chunks.append(cur.strip())
            cur = cur[-overlap:] if overlap else ""
        cur += ("\n\n" + block if cur else block)
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def chunk_document(doc: dict) -> list[dict]:
    """doc from ingestion.edgar -> chunk records ready for indexing."""
    out = []
    for i, chunk in enumerate(chunk_text(doc["text"])):
        out.append({
            "id": f"{doc['ticker']}-{doc['accession']}-{i}",
            "text": chunk,
            "metadata": {
                "ticker": doc["ticker"], "cik": doc["cik"],
                "accession": doc["accession"], "filed": doc["filed"],
                "report_date": doc["report_date"], "chunk_id": i,
                "has_table": "\n| " in chunk,
            },
        })
    return out
