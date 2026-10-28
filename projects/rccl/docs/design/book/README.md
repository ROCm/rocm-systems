# Design book build (Slate style)

`build_book.py` turns the multi-segment design notes in the parent folder into a Slate-style book: a Sphinx site on the rocm-docs-core theme with muted grey diagrams, plus a one-page HTML and a PDF.

## Build

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
python build_book.py
```

The first build downloads Mermaid 11.17.1 and rough.js 4.6.6 from jsDelivr into `_build/cache/` and checks each against a pinned SHA-256. Later builds work offline.

## Output

Everything goes to `_build/`, which git ignores:

- `html/index.html`: the website, one page per chapter.
- `rccl_multiseg_book_slate_<rev>_<stamp>.pdf`: the printed book, with bookmarks and numbered figures.
- `rccl_multiseg_book_slate_<rev>_<stamp>.check.json`: the PDF readability report.
- `rccl_multiseg_book_slate_<rev>_<stamp>.zip`: the website plus the PDF.

`<rev>` is the docs commit, with `-dirty` when the design notes have uncommitted changes.

## Checks

The build exits non-zero and lists each problem when:

- a design note in the parent folder is missing from `PARTS` in `build_book.py`;
- a diagram has no `%% caption:` line;
- a diagram's text would print below 8 pt, or the diagram doesn't fit one page;
- Sphinx reports a warning or error;
- a PDF page is wider than the print area.

## Diagrams

Each Mermaid fence becomes a numbered figure, and its `%% caption:` line becomes the caption. Diagrams are pre-rendered to SVG in headless Chromium (`mermaid_svg.py`) in the Slate light theme (`slate_theme.py`). Flowcharts use Mermaid's hand-drawn look; sequence diagrams are sketched with rough.js after rendering, because Mermaid draws them only in the plain look.

## Files

| File | Role |
| --- | --- |
| `build_book.py` | Chapter order, Sphinx project generation, PDF and zip. |
| `mermaid_svg.py` | Mermaid-to-SVG rendering, sketching, and the readability rules. |
| `slate_theme.py` | Diagram colors and fonts. |
| `make_pdf.py` | Prints the one-page HTML to PDF and checks figure text size and page overflow. |
| `mseg_slate.css` | Diagram frame and print layout on top of the rocm-docs-core theme. |
