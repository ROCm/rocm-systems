"""Render Mermaid diagrams to standalone SVG with headless Chromium (Playwright).

Used by the book build for static SVG figures and the PDF readability check.
Needs Playwright with Chromium (python -m playwright install chromium).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERMAID_VERSION = "11.17.1"
MERMAID_URL = f"https://cdn.jsdelivr.net/npm/mermaid@{MERMAID_VERSION}/dist/mermaid.min.js"
MERMAID_SHA256 = "b7a4cda9e88e187d4e1f279b8a14a3df651edd8dc704b9b56ed8e84659735570"
MERMAID_JS = HERE / "_build" / "cache" / f"mermaid-{MERMAID_VERSION}.min.js"


def _fetch_pinned(url: str, sha256: str, path: Path) -> Path:
    """Download a pinned file once into the build cache and check its hash."""
    import hashlib
    import urllib.request

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        path.unlink()
        raise SystemExit(f"{url}: sha256 {digest} != {sha256}")
    return path


def ensure_mermaid() -> Path:
    return _fetch_pinned(MERMAID_URL, MERMAID_SHA256, MERMAID_JS)


FENCE = re.compile(r"^```mermaid[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)
CAPTION = re.compile(r"^\s*%%\s*caption:\s*(.+?)\s*$", re.M)

# Layout settings that keep diagrams narrow enough to print at 8 pt or more.
READABLE = {
    "flowchart": {"useMaxWidth": True, "wrappingWidth": 170, "nodeSpacing": 26, "rankSpacing": 32, "padding": 10,
                  "htmlLabels": True},
    "sequence": {"useMaxWidth": True, "wrap": True, "width": 120, "actorMargin": 26, "messageMargin": 30,
                 "boxMargin": 8, "noteMargin": 8, "mirrorActors": False, "messageFontSize": 15,
                 "actorFontSize": 15, "noteFontSize": 14},
    "gantt": {"useMaxWidth": True, "barHeight": 26, "barGap": 6, "fontSize": 14, "sectionFontSize": 15,
              "leftPadding": 170, "topPadding": 40, "gridLineStartPadding": 30},
}


@dataclass
class Diagram:
    key: str
    src: str
    caption: str


def diagram_type(src: str) -> str:
    for line in src.splitlines():
        s = line.strip()
        if s and not s.startswith("%%"):
            return s.split()[0]
    return ""


def find_diagrams(stem: str, text: str) -> list[Diagram]:
    out = []
    for i, m in enumerate(FENCE.finditer(text)):
        src = m.group(1)
        cap = CAPTION.search(src)
        out.append(Diagram(f"{stem}-{i + 1:02d}", src, cap.group(1) if cap else ""))
    return out


def prepare_src(src: str, cfg: dict) -> str:
    """Per-diagram tweaks applied at render time only; the Markdown stays unchanged."""
    if diagram_type(src) != "sequenceDiagram":
        return src
    # Few participants get wider gaps so messages are not squeezed; many stay narrow.
    n = max(2, len(re.findall(r"^\s*(?:participant|actor)\s", src, re.M)))
    seq = cfg.get("sequence", READABLE["sequence"])
    gap = (PORTRAIT_PX * 1.25 - n * seq.get("width", 120)) / (n - 1)
    gap = int(max(seq.get("actorMargin", 26), min(gap, 260)))
    return '%%{init: {"sequence": {"actorMargin": ' + str(gap) + "}}}%%\n" + src


ROUGH_VERSION = "4.6.6"
ROUGH_URL = f"https://cdn.jsdelivr.net/npm/roughjs@{ROUGH_VERSION}/bundled/rough.js"
ROUGH_SHA256 = "6853af8619532f3939b156cd16824d2e14f31e71d3df306ad5558eeafacab194"
ROUGH_JS = HERE / "_build" / "cache" / f"rough-{ROUGH_VERSION}.js"


def ensure_rough() -> Path:
    return _fetch_pinned(ROUGH_URL, ROUGH_SHA256, ROUGH_JS)


# Mermaid has no hand-drawn sequence look; redraw its shapes with rough.js, keeping text and layout.
SKETCH_JS = r"""(svg) => {
  const rc = rough.svg(svg);
  const opt = (el) => {
    const cs = getComputedStyle(el);
    const fill = cs.fill && cs.fill !== 'none' && !/rgba\(0, 0, 0, 0\)/.test(cs.fill) ? cs.fill : undefined;
    const sw = parseFloat(cs.strokeWidth) || 1;
    const dash = cs.strokeDasharray && cs.strokeDasharray !== 'none'
      ? cs.strokeDasharray.split(/[ ,]+/).map(parseFloat) : undefined;
    return {stroke: cs.stroke === 'none' ? 'transparent' : cs.stroke, strokeWidth: sw, fill, fillStyle: 'solid',
            roughness: 1.4, bowing: 1.4, seed: 7, strokeLineDash: dash, disableMultiStroke: !!dash,
            preserveVertices: false, fillWeight: 0.6};
  };
  const swap = (el, node) => {
    for (const a of ['class', 'marker-end', 'marker-start', 'transform']) {
      const v = el.getAttribute(a); if (v) node.setAttribute(a, v);
    }
    const me = el.getAttribute('marker-end') || getComputedStyle(el).markerEnd;
    if (me && me !== 'none') {
      const p = node.querySelector('path:last-of-type'); if (p) p.setAttribute('marker-end', me);
      node.removeAttribute('marker-end');
    }
    el.replaceWith(node);
  };
  let n = 0;
  for (const el of [...svg.querySelectorAll('rect, line, path, polygon')]) {
    if (el.closest('marker, defs')) continue;
    const o = opt(el);
    let node = null;
    if (el.tagName === 'rect') {
      const w = +el.getAttribute('width'), h = +el.getAttribute('height');
      if (!w || !h) continue;
      node = rc.rectangle(+el.getAttribute('x') || 0, +el.getAttribute('y') || 0, w, h, o);
    } else if (el.tagName === 'line') {
      node = rc.line(+el.getAttribute('x1'), +el.getAttribute('y1'), +el.getAttribute('x2'), +el.getAttribute('y2'), o);
    } else if (el.tagName === 'polygon') {
      const pts = el.getAttribute('points').trim().split(/\s+/).map((p) => p.split(',').map(Number));
      node = rc.polygon(pts, o);
    } else {
      const d = el.getAttribute('d'); if (!d) continue;
      node = rc.path(d, o);
    }
    swap(el, node); n++;
  }
  return n;
}"""


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>body{margin:0;background:transparent} #out{display:inline-block}</style>
<script src="file://%s"></script><script src="file://%s"></script></head><body><div id="out"></div></body></html>"""


def font_face_css(family: str, faces: list[tuple[int, Path]]) -> str:
    """@font-face rules with the woff2 files inlined, so an SVG shown through <img> keeps the font."""
    import base64

    rules = []
    for weight, path in faces:
        data = base64.b64encode(path.read_bytes()).decode()
        rules.append(f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};"
                     f"src:url(data:font/woff2;base64,{data}) format('woff2');}}")
    return "".join(rules)


def render(diagrams: list[Diagram], cfg: dict, out_dir: Path, font_css: str = "") -> dict[str, dict]:
    """Render each diagram to out_dir/<key>.svg; return {key: {w, h, font_px}} in CSS px."""
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    page_file = out_dir / "_render.html"
    page = PAGE % (ensure_mermaid(), ensure_rough())
    if font_css:
        page = page.replace("</style>", font_css + "</style>", 1)
    page_file.write_text(page, encoding="utf-8")
    sizes: dict[str, dict] = {}
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--disable-gpu", "--no-sandbox"])
        pg = b.new_page(viewport={"width": 1600, "height": 1200})
        pg.goto(page_file.as_uri())
        if font_css:
            pg.evaluate("() => document.fonts.ready.then(() => Promise.all([...document.fonts].map(f => f.load())))")
        pg.evaluate("cfg => mermaid.initialize(Object.assign({startOnLoad:false, securityLevel:'strict'}, cfg))", cfg)
        for d in diagrams:
            src = prepare_src(d.src, cfg)
            res = pg.evaluate(
                """async ([id, src, sketch]) => {
                  const {svg} = await mermaid.render(id, src);
                  const out = document.getElementById('out'); out.innerHTML = svg;
                  const s = out.querySelector('svg'); const vb = s.viewBox.baseVal;
                  if (sketch) (new Function('return ' + sketch))()(s);
                  const t = s.querySelector('.nodeLabel, .messageText, .actor, text, span');
                  const fs = t ? parseFloat(getComputedStyle(t).fontSize) : 16;
                  return {svg: new XMLSerializer().serializeToString(s), w: vb.width, h: vb.height, font_px: fs};
                }""",
                ["m" + re.sub(r"\W", "_", d.key), src,
                 SKETCH_JS if cfg.get("look") == "handDrawn" and diagram_type(d.src) == "sequenceDiagram" else ""],
            )
            svg = res.pop("svg")
            # Standalone files need explicit dimensions; the book scales them with CSS.
            head, rest = svg.split(">", 1)
            head = re.sub(r'\s(width|height)="[^"]*"', "", head)
            svg = head.replace("<svg ", f'<svg width="{res["w"]:.0f}" height="{res["h"]:.0f}" ', 1) + ">" + rest
            svg = re.sub(r'style="max-width:[^"]*"', "", svg, count=1)
            if font_css:
                svg = re.sub(r"^(<svg[^>]*>)", lambda m: m.group(1) + f"<defs><style>{font_css}</style></defs>", svg, count=1)
            (out_dir / f"{d.key}.svg").write_text(svg, encoding="utf-8")
            sizes[d.key] = res
        b.close()
    (out_dir / "sizes.json").write_text(json.dumps(sizes, indent=1), encoding="utf-8")
    return sizes


# Letter paper, 0.75 in side margins and room for a caption: text box in CSS px.
PORTRAIT_PX, LANDSCAPE_PX = 7.0 * 96, 9.5 * 96
PORTRAIT_H_PX, LANDSCAPE_H_PX = 8.6 * 96, 6.1 * 96
# The printed text box is 7.2 in (0.65 in margins); a figure keeps 7.0 in after its surface padding.
PRINT_BOX_PX = 7.2 * 96
MIN_PT = 8.0


def effective_pt(w: float, font_px: float, page_px: float, h: float = 0.0, page_h_px: float = 1e9) -> float:
    return font_px * min(1.0, page_px / max(w, 1.0), page_h_px / max(h, 1.0)) * 0.75


def placement(w: float, h: float, font_px: float) -> str:
    """portrait, landscape, or too-big (needs a source change); a figure is scaled to fit one page."""
    if effective_pt(w, font_px, PORTRAIT_PX, h, PORTRAIT_H_PX) >= MIN_PT:
        return "portrait"
    if effective_pt(w, font_px, LANDSCAPE_PX, h, LANDSCAPE_H_PX) >= MIN_PT:
        return "landscape"
    return "too-big"
