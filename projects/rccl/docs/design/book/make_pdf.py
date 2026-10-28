"""Print a single-page HTML book to PDF and check every diagram is readable.

python make_pdf.py <html> <out.pdf> --title T --rev R [--fig-selector CSS] [--pages-dir DIR]

Uses Chromium's print engine (bookmarks from headings, tagged PDF, header and
footer), then measures each figure in print layout: effective font size in pt
must be at least mermaid_svg.MIN_PT. Pages that hold a figure are rendered to
PNG under --pages-dir for visual review.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mermaid_svg as M  # noqa: E402  (sets PLAYWRIGHT_BROWSERS_PATH)

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("pdf")
    ap.add_argument("--title", required=True)
    ap.add_argument("--rev", default="")
    ap.add_argument("--fig-selector", default="figure img, figure pre.mermaid svg")
    ap.add_argument("--pages-dir", default="")
    ap.add_argument("--wait-selector", default="")
    ap.add_argument("--wait-all", default="", help="wait until every element matching this selector holds an svg")
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright

    html = Path(a.html).resolve()
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
        pg.pdf(path=a.pdf, format="Letter", print_background=True, prefer_css_page_size=True,
               display_header_footer=True, outline=True, tagged=True,
               header_template=HEADER.format(title=a.title, rev=a.rev), footer_template=FOOTER,
               margin={"top": "0.7in", "bottom": "0.7in", "left": "0.65in", "right": "0.65in"})
        b.close()

    low = [f for f in figs if round(f["pt"], 1) < M.MIN_PT]
    report = {"pdf": a.pdf, "figures": len(figs), "below_min_pt": len(low), "min_pt": M.MIN_PT,
              "overflow_px": max(overflow, 0),
              "smallest": sorted(figs, key=lambda f: f["pt"])[:8]}
    import fitz

    doc = fitz.open(a.pdf)
    report["pages"] = doc.page_count
    report["bookmarks"] = len(doc.get_toc())
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
