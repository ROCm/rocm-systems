# Design and code walkthrough books (Slate style)

`build_book.py` turns the multi-segment docs into two Slate-style books, each a Sphinx site on the rocm-docs-core theme with muted grey diagrams, plus a one-page HTML and a PDF:

- the design book, "RCCL Multi-Segment DMA-BUF Design", from the design notes in the parent folder;
- the code walkthrough book, "RCCL Multi-Segment DMA-BUF Code Walkthrough", from `walkthrough/`, one part per fork.

## Build

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
python build_book.py                      # both books
python build_book.py --book design        # or --book walkthrough
```

The first build downloads Mermaid 11.17.1 and rough.js 4.6.6 from jsDelivr into `_build/cache/` and checks each against a pinned SHA-256. Later builds work offline.

## Output

Each book goes to `_build/<book>/`, which git ignores, where `<book>` is `design` or `walkthrough`:

- `html/index.html`: the website, one page per chapter.
- `rccl_multiseg_<book>_<rev>_<stamp>.pdf`: the printed book, with a Contents listing parts, chapters and sections with page numbers, bookmarks, and numbered figures.
- `rccl_multiseg_<book>_<rev>_<stamp>.check.json`: the PDF readability report.
- `rccl_multiseg_<book>_<rev>_<stamp>.zip`: the website plus the PDF.

`<rev>` is the docs commit, with `-dirty` when the docs have uncommitted changes.

Links from one book to a doc of the other book point at its copy under `DOCS_URL`, and links to source files point under `CODE_URL`; both are set in `build_book.py`.

## Checks

The build exits non-zero and lists each problem when:

- a doc in a book's folder is missing from that book's `parts` in `build_book.py`, or a listed doc does not exist;
- a diagram has no `%% caption:` line;
- a diagram's text would print below 8 pt, or the diagram doesn't fit one page;
- Sphinx reports a warning or error;
- a PDF page is wider than the print area.

## Contents and bookmarks

Chromium names a PDF bookmark after every layout fragment of a heading, so a heading moved to the next page shows up twice ("3.2. 3.2. SelectionSelection") and a wrapped title loses the space at the break. `make_pdf.py` therefore writes the Contents and the bookmarks from the HTML headings: it prints once to learn each heading's page, fills the Contents with those pages, prints again until the pages are stable, then replaces the bookmarks with one entry per part (on its first chapter's page), chapter and section.

## Diagrams

Each Mermaid fence becomes a numbered figure, and its `%% caption:` line becomes the caption. Diagrams are pre-rendered to SVG in headless Chromium (`mermaid_svg.py`) in the Slate light theme (`slate_theme.py`). Flowcharts use Mermaid's hand-drawn look; sequence diagrams are sketched with rough.js after rendering, because Mermaid draws them only in the plain look.

## Files

| File | Role |
| --- | --- |
| `build_book.py` | Book configs and chapter order, Sphinx project generation, PDF and zip. |
| `mermaid_svg.py` | Mermaid-to-SVG rendering, sketching, and the readability rules. |
| `slate_theme.py` | Diagram colors and fonts. |
| `make_pdf.py` | Prints the one-page HTML to PDF, writes the Contents and bookmarks, and checks figure text size and page overflow. |
| `mseg_slate.css` | Diagram frame, Contents and print layout on top of the rocm-docs-core theme. |
