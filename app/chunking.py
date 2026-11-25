import re
from dataclasses import dataclass

from app.parsing import Page

# "# Title" or a short line that is only bold: "**Research and Development**"
MD_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
BOLD_HEADING = re.compile(r"^\*\*([^*]{3,100})\*\*$")


@dataclass
class ChunkData:
    chunk_index: int
    page: int
    section: str | None
    text: str


def heading_of(block: str) -> str | None:
    first = block.split("\n", 1)[0].strip()
    m = MD_HEADING.match(first) or BOLD_HEADING.match(first)
    if not m:
        return None
    title = re.sub(r"[*_`#]+", "", m.group(1)).strip()
    # "For the years ended December 31," or "(in thousands)" are captions, not titles
    if title.endswith((",", ":", ";")) or title.startswith("("):
        return None
    if not re.search(r"[A-Za-z]{3}", title):  # "## 22" is a page number, not a title
        return None
    return title if len(title.split()) <= 12 else None


def split_long(block: str, max_words: int, overlap: int) -> list[str]:
    words = block.split()
    step = max_words - overlap
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), step)
            if words[i:i + max_words]]


def chunk_pages(pages: list[Page], filename: str, max_words: int = 300,
                overlap: int = 40, min_words: int = 30) -> list[ChunkData]:
    chunks: list[ChunkData] = []
    section: str | None = None

    def emit(page_no: int, body: str):
        # a tiny piece (cover text, a map label) joins the previous chunk of the same page
        if len(body.split()) < min_words and chunks and chunks[-1].page == page_no:
            chunks[-1].text += "\n\n" + body
            return
        header = f"[{filename} | {section}]" if section else f"[{filename}]"
        chunks.append(ChunkData(len(chunks), page_no, section, f"{header}\n{body}"))

    for page in pages:
        buf: list[str] = []
        buf_words = 0
        has_body = False  # True once buf holds more than headings
        for block in re.split(r"\n\s*\n", page.markdown):
            block = block.strip()
            if not block:
                continue
            title = heading_of(block)
            if title:
                if has_body:  # close the previous section's text
                    emit(page.number, "\n\n".join(buf))
                    buf, buf_words, has_body = [], 0, False
                section = title
                buf.append(block)  # the heading stays with the text under it
                buf_words += len(block.split())
                continue
            n = len(block.split())
            if n > max_words:
                if has_body:
                    emit(page.number, "\n\n".join(buf))
                    buf, buf_words, has_body = [], 0, False
                for piece in split_long(block, max_words, overlap):
                    emit(page.number, piece)
                continue
            if buf_words + n > max_words and has_body:
                emit(page.number, "\n\n".join(buf))
                buf, buf_words = [], 0
            buf.append(block)
            buf_words += n
            has_body = True
        if buf:
            emit(page.number, "\n\n".join(buf))
    return chunks