import re
import threading
from dataclasses import dataclass

import pymupdf
import pymupdf4llm


@dataclass
class Page:
    number: int      # 1-based, eval.py expects this
    markdown: str


# PyMuPDF is not thread-safe: two PDFs parsed at the same time in different
# threads break each other ("not a textpage of this page"). One at a time.
_parse_lock = threading.Lock()


def parse_pdf(path: str) -> list[Page]:
    with _parse_lock:
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