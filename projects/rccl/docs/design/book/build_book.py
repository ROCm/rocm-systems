#!/usr/bin/env python3
"""Build the multi-segment design notes as an AMD-style (rocm-docs-core) book and PDF.

    python -m pip install -r requirements.txt && python -m playwright install chromium
    python build_book.py

Everything is written under _build/: the generated Sphinx project (src/), the site (html/),
the one-page build that is printed (single/), the PDF, its readability report, and a zip.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import amd_theme as A
import mermaid_svg as M

HERE = Path(__file__).resolve().parent
DESIGN = HERE.parent
BUILD = HERE / "_build"
SRC, HTML, SINGLE = BUILD / "src", BUILD / "html", BUILD / "single"
TITLE = "RCCL Multi-Segment DMA-BUF Design"
REPO_URL = "https://github.com/ROCm/rocm-systems"
CODE_URL = f"{REPO_URL}/blob/users/atulkulk/mseg/multi-segment-dmabuf-cast-gin/projects/rccl/"

PARTS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Part I — Overview", [
        ("multi_segment_overview.md", "Overview (start here)"),
        ("multi_segment_current.md", "Current behavior and invariants"),
        ("multi_segment_sync_and_removal.md", "Upstream sync impact and removal plan"),
    ]),
    ("Part II — Design per transport layer", [
        ("multi_segment_gin_rma.md", "Classic GIN / RMA"),
        ("multi_segment_netib_wire.md", "Classic NET/IB P2P"),
        ("multi_segment_netib_cast.md", "IB-CAST NET/IB P2P"),
        ("multi_segment_netib_cast_gin.md", "IB-CAST GIN / RMA"),
    ]),
    ("Part III — Testing and related designs", [
        ("multi_segment_testing.md", "Testing and validation"),
        ("multi_segment_gin_rocshmem_gda_and_anvil_sdma.md", "rocSHMEM GDA and Anvil SDMA (proposed)"),
    ]),
    ("Part IV — Follow-up designs", [
        ("multi_segment_varseg.md", "Register past 16 segments"),
        ("multi_segment_sidetable_first_reg.md", "Side table on first registration"),
        ("multi_segment_flush_read_collapse.md", "Flush read collapse"),
    ]),
    ("Part V — Reference", [
        ("multi_segment_code_map.md", "Code map"),
    ]),
]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(DESIGN), *args], capture_output=True, text=True, check=True).stdout.strip()


def docs_rev() -> str:
    return git("rev-parse", "--short=12", "HEAD") + ("-dirty" if git("status", "--porcelain", "--", ".") else "")


ALERT = re.compile(r"^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*\n((?:>.*\n?)*)", re.M)


def alerts_to_admonitions(text: str) -> str:
    def sub(m):
        kind = m.group(1).lower()
        body = "\n".join(re.sub(r"^> ?", "", l) for l in m.group(2).rstrip("\n").split("\n"))
        return f":::{{{kind}}}\n{body.strip()}\n:::\n"
    return ALERT.sub(sub, text)


def code_links(text: str) -> str:
    """Repo-relative source links point at the pushed wire branch on GitHub."""
    def sub(m):
        path = re.sub(r"^(\.\./)+", "", m.group(2))
        return f"[{m.group(1)}]({CODE_URL}{path})"
    return re.sub(r"\[([^\]]+)\]\(((?:\.\./)+(?:src|test|tools|cmake)[^)\s]*)\)", sub, text)


def main() -> int:
    rev = docs_rev()
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    if SRC.exists():
        shutil.rmtree(SRC)
    (SRC / "_diagrams").mkdir(parents=True)
    (SRC / "_static").mkdir()

    files = [DESIGN / f for _, docs in PARTS for f, _ in docs]
    problems = [f"design doc not in book: {p.name}" for p in sorted(DESIGN.glob("*.md")) if p not in files]
    all_diagrams: list[M.Diagram] = []
    for fname in files:
        stem = fname.stem.replace("multi_segment_", "")
        for d in M.find_diagrams(stem, fname.read_text()):
            all_diagrams.append(M.Diagram(d.key, A.recolor(d.src), d.caption))
    sizes = M.render(all_diagrams, A.CFG, SRC / "_diagrams", A.FONT_CSS)
    (SRC / "_diagrams" / "_render.html").unlink()

    placements = {}
    for d in all_diagrams:
        s = sizes[d.key]
        placements[d.key] = M.placement(s["w"], s["h"], s["font_px"])
        if not d.caption:
            problems.append(f"{d.key}: no %% caption")
        if placements[d.key] == "too-big":
            problems.append(f"{d.key}: too big for one page at {M.MIN_PT} pt")

    for fname in files:
        stem = fname.stem.replace("multi_segment_", "")
        n = [0]

        def fig(m):
            n[0] += 1
            key = f"{stem}-{n[0]:02d}"
            cap = M.CAPTION.search(m.group(1))
            cap = cap.group(1) if cap else key
            cls = "amd-diagram" + (" fig-landscape" if placements[key] == "landscape" else "")
            return f"```{{figure}} _diagrams/{key}.svg\n:name: fig-{key}\n:alt: {cap}\n:figclass: {cls}\n\n{cap}\n```"

        text = M.FENCE.sub(fig, fname.read_text())
        text = code_links(alerts_to_admonitions(text))
        for (lf, ls), (df, ds) in A.CLASS_MAP.items():
            text = text.replace(f"`{ls}` | `{lf}`", f"`{ds}` | `{df}`")
        (SRC / fname.name).write_text(text)

    toc = ["defaults:\n  numbered: true\n  maxdepth: 2\nroot: index\nsubtrees:"]
    for part, docs in PARTS:
        toc.append(f"- caption: \"{part}\"\n  entries:")
        toc += [f"  - file: {Path(f).stem}" for f, _ in docs]
    (SRC / ".sphinx").mkdir()
    (SRC / ".sphinx" / "_toc.yml.in").write_text("\n".join(toc) + "\n")
    (SRC / "index.md").write_text(
        f"# {TITLE}\n\nDesign notes for registering one DMA-BUF memory region per physical segment on every "
        f"InfiniBand path in RCCL. Built from docs commit `{rev}` on {stamp[:8]}.\n\n"
        "Start with the overview. Each part can be read on its own; figures are numbered per chapter.\n")
    shutil.copy(HERE / "mseg_amd.css", SRC / "_static" / "mseg_amd.css")
    (SRC / "conf.py").write_text(CONF.format(title=TITLE, rev=rev, repo=REPO_URL, branch=branch))

    for builder, out in (("html", HTML), ("singlehtml", SINGLE)):
        log = BUILD / f"sphinx_{builder}.log"
        with log.open("w") as fh:
            r = subprocess.run([sys.executable, "-m", "sphinx", "-b", builder, "-q", "-E", str(SRC), str(out)],
                               stdout=fh, stderr=subprocess.STDOUT)
        warns = [l for l in log.read_text().splitlines() if ("WARNING" in l or "ERROR" in l)
                 and "Not in a Git Directory" not in l and "failed to reach any of the inventories" not in l]
        print(f"sphinx {builder}: rc={r.returncode} warnings={len(warns)} log={log}")
        problems += [f"{builder}: {w}" for w in warns[:20]]
        if r.returncode:
            return r.returncode

    pdf = BUILD / f"rccl_multiseg_book_amd_{rev}_{stamp}.pdf"
    rc = subprocess.run([sys.executable, str(HERE / "make_pdf.py"), str(SINGLE / "index.html"), str(pdf),
                         "--title", TITLE, "--rev", f"docs {rev}", "--pages-dir", str(BUILD / "pdf_pages"),
                         "--fig-selector", "figure.amd-diagram img"]).returncode
    if rc:
        problems.append("PDF readability check failed; see " + str(pdf.with_suffix(".check.json")))

    zpath = BUILD / f"rccl_multiseg_book_amd_{rev}_{stamp}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(HTML.rglob("*")):
            if p.is_file():
                z.write(p, f"html/{p.relative_to(HTML)}")
        if pdf.exists():
            z.write(pdf, pdf.name)
        z.writestr("README.txt", f"{TITLE} (AMD style)\nDocs commit {rev}, built {stamp}\n\n"
                   "Open html/index.html, or read the PDF.\n")
    (BUILD / "placements.json").write_text(json.dumps(placements, indent=1))
    print("html:", HTML / "index.html")
    print("pdf:", pdf)
    print("zip:", zpath, zpath.stat().st_size, "bytes")
    print(f"problems: {len(problems)}")
    for p in problems:
        print("  -", p)
    return 1 if problems else 0


CONF = '''import rocm_docs.util
from rocm_docs import ROCmDocs

# The generated source is not a git checkout; name the branch the notes come from.
rocm_docs.util.get_branch = lambda *a, **k: ("{repo}", "{branch}")

docs_core = ROCmDocs("{title}")
docs_core.setup()
for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

project = "{title}"
version = release = "docs {rev}"
html_theme_options = {{"flavor": "rocm", "link_main_doc": False, "repository_provider": "github",
                      "use_repository_button": False, "show_toc_level": 2}}
html_static_path = ["_static"]
html_css_files = ["mseg_amd.css"]
numfig = True
numfig_format = {{"figure": "Figure %s"}}
numfig_secnum_depth = 1
myst_heading_anchors = 4
nb_execution_mode = "off"
external_projects = []
external_projects_current_project = "rccl"
exclude_patterns = ["_build", "_diagrams/sizes.json"]
suppress_warnings = ["myst.header", "toc.not_included"]
'''

if __name__ == "__main__":
    sys.exit(main())
