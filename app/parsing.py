import re
from dataclasses import dataclass

import pymupdf
import pymupdf4llm


@dataclass
class Page:
    number: int      # 1-based, eval.py belə gözləyir
    markdown: str


def parse_pdf(path: str) -> list[Page]:
    doc = pymupdf.open(path)
    try:
        parts = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
    finally:
        doc.close()

    pages = []
    for i, part in enumerate(parts):
        text = (part.get("text") or "").strip()
        text = re.sub(r"\n*-{5,}\s*$", "", text).strip()
        pages.append(Page(number=i + 1, markdown=text))
    return pages