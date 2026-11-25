import sys

from app.chunking import chunk_pages
from app.parsing import parse_pdf

path = sys.argv[1]
name = path.split("/")[-1]
chunks = chunk_pages(parse_pdf(path), name)
print(f"{len(chunks)} chunks")
for c in chunks:
    if c.page == int(sys.argv[2]):
        print("-" * 60)
        print(f"page {c.page} | section: {c.section} | {len(c.text.split())} words")
        print(c.text[:500])