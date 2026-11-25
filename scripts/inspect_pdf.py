import sys

from app.parsing import parse_pdf

path, page_no = sys.argv[1], int(sys.argv[2])
pages = parse_pdf(path)
print(f"Total pages: {len(pages)}\n")
print(pages[page_no - 1].markdown)