"""Print a single-page HTML book to PDF and check every diagram is readable.

python make_pdf.py <html> <out.pdf> --title T --rev R [--parts JSON] [--fig-selector CSS] [--pages-dir DIR]

Uses Chromium's print engine (tagged PDF, header and footer), then measures each figure in print
layout: effective font size in pt must be at least mermaid_svg.MIN_PT. Pages that hold a figure are
rendered to PNG under --pages-dir for visual review.

Contents and bookmarks are built here, not by Chromium: Chromium names an outline entry by joining
the text of every fragment of a heading, so a heading pushed to the next page by break-after: avoid
reads "3.2. 3.2. SelectionSelection" and a wrapped title loses the space at the line break. The
page is printed once with placeholder page numbers, the heading pages are read from that PDF, and
the Contents is filled in and printed again; the bookmarks are then rewritten from the DOM headings,
with each Part on its first chapter's page and its chapters nested under it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mermaid_svg as M  # noqa: E402  (sets PLAYWRIGHT_BROWSERS_PATH)

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF
    import fitz

HEADER = ('<div style="font-family:Open Sans,Arial,sans-serif;font-size:7.5pt;color:#6b6b70;width:100%;'
          'padding:0 0.65in;display:flex;justify-content:space-between">'
          '<span>{title}</span><span>{rev}</span></div>')
FOOTER = ('<div style="font-family:Open Sans,Arial,sans-serif;font-size:7.5pt;color:#6b6b70;width:100%;'
          'padding:0 0.65in;display:flex;justify-content:space-between">'
          '<span class="title"></span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')

MEASURE = """(sel) => {
  const out = [];
  document.querySelectorAll(sel).forEach((el, i) => {
    const r0 = el.getBoundingClientRect(), cs = getComputedStyle(el);
    const r = {width: r0.width - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight), height: r0.height};
    let natW = 0, fontPx = 16;
    if (el.tagName === 'IMG') { natW = el.naturalWidth; fontPx = parseFloat(el.dataset.fontPx || '16'); }
    else { const s = el.tagName === 'svg' ? el : el.querySelector('svg'); if (!s) return;
           natW = s.viewBox.baseVal.width; const t = s.querySelector('.nodeLabel, .messageText, text');
           fontPx = t ? parseFloat(getComputedStyle(t).fontSize) : 16; }
    const fig = el.closest('figure');
    const cap = fig ? (fig.querySelector('figcaption') || {}).textContent || '' : '';
    out.push({i, w: r.width, h: r.height, natW, fontPx, pt: fontPx * (r.width / Math.max(natW, 1)) * 0.75,
              caption: cap.replace(/\\s+/g, ' ').trim().slice(0, 90)});
  });
  return out;
}"""

# Every visible heading in document order, as Chromium puts them in the PDF outline.
HEADINGS = """() => [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')]
  .filter((h) => h.getClientRects().length > 0)
  .map((h) => { const c = h.cloneNode(true); c.querySelectorAll('.headerlink').forEach((a) => a.remove());
                const s = h.closest('section');
                return {level: +h.tagName[1], text: c.textContent.replace(/\\s+/g, ' ').trim(),
                        id: (s && s.id) || h.id || ''}; })"""

# Replace the theme's print TOC (chapter titles only, Part captions as ARIA headings) with a
# Contents of Parts, chapters and sections; no heading elements, so the outline stays clean.
FILL_TOC = """([rows, title]) => {
  const box = document.querySelector('#jb-print-toc');
  if (!box) return false;
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const body = rows.map((r) => r.kind === 'part'
      ? `<div class="mseg-toc-part">${esc(r.text)}</div>`
      : `<a class="mseg-toc-row mseg-toc-${r.kind}" href="#${r.id}"><span class="mseg-toc-t">${esc(r.text)}</span>`
        + `<span class="mseg-toc-dots"></span><span class="mseg-toc-p">${r.page}</span></a>`).join('');
  box.innerHTML = `<div class="mseg-toc-title">${esc(title)}</div><div class="mseg-toc">${body}</div>`;
  return true;
}"""


def print_pdf(pg, path: str, a) -> None:
    pg.pdf(path=path, format="Letter", print_background=True, prefer_css_page_size=True,
           display_header_footer=True, outline=True, tagged=True,
           header_template=HEADER.format(title=a.title, rev=a.rev), footer_template=FOOTER,
           margin={"top": "0.7in", "bottom": "0.7in", "left": "0.65in", "right": "0.65in"})


def heading_pages(pdf: str, heads: list[dict]) -> list[int]:
    """Page of each DOM heading, from Chromium's outline, which lists the same headings in order."""
    toc = fitz.open(pdf).get_toc(simple=False)
    if len(toc) != len(heads):
        raise SystemExit(f"outline has {len(toc)} entries but the page has {len(heads)} headings")
    return [e[2] for e in toc]


def contents_rows(heads, pages, parts):
    rows, ch, pi = [], 0, -1
    bounds = []  # chapter index at which each part starts
    n = 0
    for _, count in parts:
        bounds.append(n)
        n += count
    for h, p in zip(heads, pages):
        if h["level"] == 2:
            while pi + 1 < len(bounds) and ch >= bounds[pi + 1]:
                pi += 1
                rows.append({"kind": "part", "text": parts[pi][0]})
            ch += 1
            rows.append({"kind": "chapter", "text": h["text"], "id": h["id"], "page": p})
        elif h["level"] == 3 and ch:
            rows.append({"kind": "section", "text": h["text"], "id": h["id"], "page": p})
    return rows


def bookmarks(title, heads, pages, parts, contents_page):
    toc = [[1, title, 1], [1, "Contents", contents_page]]
    base = 1 if parts else 0
    bounds, n = [], 0
    for _, count in parts:
        bounds.append(n)
        n += count
    ch, pi = 0, -1
    for h, p in zip(heads, pages):
        if h["level"] < 2:
            continue
        if h["level"] == 2:
            while pi + 1 < len(bounds) and ch >= bounds[pi + 1]:
                pi += 1
                toc.append([1, parts[pi][0], p])
            ch += 1
        if not ch:
            continue
        level = h["level"] - 1 + base
        toc.append([min(level, toc[-1][0] + 1), h["text"], p])
    return toc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("pdf")
    ap.add_argument("--title", required=True)
    ap.add_argument("--rev", default="")
    ap.add_argument("--parts", default="[]", help='JSON list of [part title, number of chapters]')
    ap.add_argument("--fig-selector", default="figure img, figure pre.mermaid svg")
    ap.add_argument("--pages-dir", default="")
    ap.add_argument("--wait-selector", default="")
    ap.add_argument("--wait-all", default="", help="wait until every element matching this selector holds an svg")
    a = ap.parse_args()
    parts = json.loads(a.parts)

    from playwright.sync_api import sync_playwright

    html = Path(a.html).resolve()
    draft = str(Path(a.pdf).with_suffix(".draft.pdf"))
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--disable-gpu", "--no-sandbox"])
        # Print text box is 7.0 in wide at 96 CSS px per inch.
        pg = b.new_page(viewport={"width": int(M.PRINT_BOX_PX), "height": 1000})
        pg.emulate_media(media="print")
        pg.goto(html.as_uri(), wait_until="load")
        if a.wait_selector:
            pg.wait_for_selector(a.wait_selector, timeout=180000)
        if a.wait_all:
            pg.wait_for_function("(s) => { const e = document.querySelectorAll(s); return e.length > 0 && "
                                 "[...e].every((x) => x.querySelector('svg')); }", arg=a.wait_all, timeout=300000)
        pg.evaluate("() => document.fonts.ready")
        pg.wait_for_timeout(1500)
        figs = pg.evaluate(MEASURE, a.fig_selector)
        # Content wider than the page makes Chromium shrink every page to fit.
        overflow = pg.evaluate("() => document.documentElement.scrollWidth") - int(M.PRINT_BOX_PX)

        # Placeholder numbers are as wide as real ones, so the layout does not move between passes.
        pg.evaluate(FILL_TOC, [[], "Contents"])
        heads = pg.evaluate(HEADINGS)
        pages = [999] * len(heads)
        for _ in range(3):
            pg.evaluate(FILL_TOC, [contents_rows(heads, pages, parts), "Contents"])
            print_pdf(pg, draft, a)
            new = heading_pages(draft, heads)
            if new == pages:
                break
            pages = new
        os.replace(draft, a.pdf)
        b.close()

    doc = fitz.open(a.pdf)
    contents_page = next((i + 1 for i in range(min(5, doc.page_count)) if "Contents" in doc[i].get_text()), 1)
    toc = bookmarks(a.title, heads, pages, parts, contents_page)
    doc.set_toc(toc)
    tmp = a.pdf + ".tmp"
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    os.replace(tmp, a.pdf)

    low = [f for f in figs if round(f["pt"], 1) < M.MIN_PT]
    report = {"pdf": a.pdf, "figures": len(figs), "below_min_pt": len(low), "min_pt": M.MIN_PT,
              "overflow_px": max(overflow, 0),
              "smallest": sorted(figs, key=lambda f: f["pt"])[:8]}
    doc = fitz.open(a.pdf)
    report["pages"] = doc.page_count
    report["bookmarks"] = len(doc.get_toc())
    first_body = min([p for h, p in zip(heads, pages) if h["level"] >= 2] or [contents_page + 1])
    report["front_matter_pages"] = [1, first_body - 1]
    if a.pages_dir:
        out = Path(a.pages_dir)
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for page in doc:
            if page.get_images() or len(page.get_drawings()) > 60:
                page.get_pixmap(dpi=100).save(out / f"page_{page.number + 1:03d}.png")
                n += 1
        report["figure_pages_rendered"] = n
    Path(a.pdf).with_suffix(".check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "smallest"}))
    for f in report["smallest"]:
        print(f"  {f['pt']:5.1f} pt  {f['caption']}")
    return 1 if low or overflow > 2 else 0


if __name__ == "__main__":
    sys.exit(main())
