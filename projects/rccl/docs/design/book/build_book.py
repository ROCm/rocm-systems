#!/usr/bin/env python3
"""Build the multi-segment design book and the code walkthrough book (Slate style, rocm-docs-core) and PDFs.

    python -m pip install -r requirements.txt && python -m playwright install chromium
    python build_book.py [--book design|walkthrough|both]

Each book is written under _build/<book>/: the generated Sphinx project (src/), the site (html/),
the one-page build that is printed (single/), the PDF, its readability report, and a zip.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import slate_theme as A
import mermaid_svg as M

HERE = Path(__file__).resolve().parent
DESIGN = HERE.parent
REPO_URL = "https://github.com/ROCm/rocm-systems"
# Code links and links between the two books point here; both need the branch to be on GitHub.
CODE_URL = f"{REPO_URL}/blob/users/atulkulk/netib-mseg/04-rma-gin-multiseg/projects/rccl/"
DOCS_URL = f"{REPO_URL}/blob/develop/projects/rccl/docs/design/"

BOOKS: dict[str, dict] = {
    "design": dict(
        title="RCCL Multi-Segment DMA-BUF Design",
        folder=DESIGN,
        intro="Design notes for registering one DMA-BUF memory region per physical segment on every "
              "InfiniBand path in RCCL. The code paths are in the separate code walkthrough book.",
        parts=[
            ("Part I — Overview", [
                ("multi_segment_overview.md", "Overview (start here)"),
                ("multi_segment_current.md", "Current contract"),
            ]),
            ("Part II — Design", [
                ("multi_segment_fork_plugin.md", "IB-MSEG plugin and the fork stack"),
                ("multi_segment_gin_rma.md", "GIN / RMA (core and IB-MSEG)"),
                ("multi_segment_netib_wire.md", "IB-MSEG NET/IB P2P"),
                ("multi_segment_netib_cast.md", "IB-CAST NET/IB P2P (fork 6 design)"),
                ("multi_segment_netib_cast_gin.md", "IB-CAST GIN / RMA (fork 6 design)"),
                ("multi_segment_gin_rocshmem_gda_and_anvil_sdma.md", "rocSHMEM GDA and Anvil SDMA (proposed)"),
            ]),
            ("Part III — Testing", [
                ("multi_segment_testing.md", "Testing"),
                ("multi_segment_validation_plan.md", "Validation plan"),
            ]),
            ("Part IV — Reference", [
                ("multi_segment_code_map.md", "Code map"),
                ("multi_segment_review_feedback.md", "Review feedback"),
            ]),
        ]),
    "walkthrough": dict(
        title="RCCL Multi-Segment DMA-BUF Code Walkthrough",
        folder=DESIGN / "walkthrough",
        intro="Code walkthroughs of the IB-MSEG fork stack, one part per fork, each following the code in "
              "execution order. The design rationale is in the separate design book.",
        parts=[
            ("Part I — Fork 1: core gate", [("multi_segment_walkthrough_fork1_core.md", "Fork 1")]),
            ("Part II — Fork 2: net_ib copy", [("multi_segment_walkthrough_fork2_copy.md", "Fork 2")]),
            ("Part III — Fork 3: registration and P2P", [("multi_segment_walkthrough_fork3_p2p.md", "Fork 3")]),
            ("Part IV — Fork 4: RMA and GIN", [("multi_segment_walkthrough_fork4_gin_rma.md", "Fork 4")]),
        ]),
}


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


def book_links(text: str, folder: Path, in_book: set[Path]) -> str:
    """Links to a doc of the other book point at its GitHub copy; Sphinx sees only this book."""
    def sub(m):
        target = (folder / m.group(2)).resolve()
        if target in in_book or not target.is_relative_to(DESIGN):
            return m.group(0)
        return f"[{m.group(1)}]({DOCS_URL}{target.relative_to(DESIGN).as_posix()}{m.group(3) or ''})"
    return re.sub(r"\[([^\]]+)\]\(([^)\s#:]+\.md)(#[^)\s]*)?\)", sub, text)


def code_links(text: str) -> str:
    """Repo-relative source links point at the pushed wire branch on GitHub."""
    def sub(m):
        path = re.sub(r"^(\.\./)+", "", m.group(2))
        return f"[{m.group(1)}]({CODE_URL}{path})"
    return re.sub(r"\[([^\]]+)\]\(((?:\.\./)+(?:src|test|tools|cmake)[^)\s]*)\)", sub, text)


def build(book: str) -> int:
    cfg = BOOKS[book]
    title, folder, parts = cfg["title"], cfg["folder"], cfg["parts"]
    build_dir = HERE / "_build" / book
    src, html, single = build_dir / "src", build_dir / "html", build_dir / "single"
    rev = docs_rev()
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    if src.exists():
        shutil.rmtree(src)
    (src / "_diagrams").mkdir(parents=True)
    (src / "_static").mkdir()

    listed = [folder / f for _, docs in parts for f, _ in docs]
    problems = [f"{book}: doc in {folder.name}/ not in the book: {p.name}"
                for p in sorted(folder.glob("*.md")) if p not in listed and p.name != "README.md"]
    problems += [f"{book}: missing doc {p.name}" for p in listed if not p.exists()]
    files = [p for p in listed if p.exists()]
    parts = [(part, [(f, n) for f, n in docs if (folder / f).exists()]) for part, docs in parts]
    parts = [(part, docs) for part, docs in parts if docs]
    in_book = {p.resolve() for p in files}
    all_diagrams: list[M.Diagram] = []
    for fname in files:
        stem = fname.stem.replace("multi_segment_", "")
        for d in M.find_diagrams(stem, fname.read_text()):
            all_diagrams.append(M.Diagram(d.key, A.recolor(d.src), d.caption))
    sizes = M.render(all_diagrams, A.CFG, src / "_diagrams", A.FONT_CSS)
    (src / "_diagrams" / "_render.html").unlink()

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
            cls = "slate-diagram" + (" fig-landscape" if placements[key] == "landscape" else "")
            return f"```{{figure}} _diagrams/{key}.svg\n:name: fig-{key}\n:alt: {cap}\n:figclass: {cls}\n\n{cap}\n```"

        text = M.FENCE.sub(fig, fname.read_text())
        text = book_links(code_links(alerts_to_admonitions(text)), folder, in_book)
        for (lf, ls), (df, ds) in A.CLASS_MAP.items():
            text = text.replace(f"`{ls}` | `{lf}`", f"`{ds}` | `{df}`")
        (src / fname.name).write_text(text)

    toc = ["defaults:\n  numbered: true\n  maxdepth: 2\nroot: index\nsubtrees:"]
    for part, docs in parts:
        toc.append(f"- caption: \"{part}\"\n  entries:")
        toc += [f"  - file: {Path(f).stem}" for f, _ in docs]
    (src / ".sphinx").mkdir()
    (src / ".sphinx" / "_toc.yml.in").write_text("\n".join(toc) + "\n")
    (src / "index.md").write_text(
        f"# {title}\n\n{cfg['intro']} Built from docs commit `{rev}` on {stamp[:8]}.\n\n"
        "Each part can be read on its own; figures are numbered per chapter.\n")
    shutil.copy(HERE / "mseg_slate.css", src / "_static" / "mseg_slate.css")
    (src / "conf.py").write_text(CONF.format(title=title, rev=rev, repo=REPO_URL, branch=branch))

    for builder, out in (("html", html), ("singlehtml", single)):
        log = build_dir / f"sphinx_{builder}.log"
        with log.open("w") as fh:
            r = subprocess.run([sys.executable, "-m", "sphinx", "-b", builder, "-q", "-E", str(src), str(out)],
                               stdout=fh, stderr=subprocess.STDOUT)
        warns = [l for l in log.read_text().splitlines() if ("WARNING" in l or "ERROR" in l)
                 and "Not in a Git Directory" not in l and "failed to reach any of the inventories" not in l]
        print(f"{book} sphinx {builder}: rc={r.returncode} warnings={len(warns)} log={log}")
        problems += [f"{book} {builder}: {w}" for w in warns[:20]]
        if r.returncode:
            return r.returncode

    name = f"rccl_multiseg_{book}_{rev}_{stamp}"
    pdf = build_dir / f"{name}.pdf"
    part_counts = json.dumps([[part, len(docs)] for part, docs in parts])
    rc = subprocess.run([sys.executable, str(HERE / "make_pdf.py"), str(single / "index.html"), str(pdf),
                         "--title", title, "--rev", f"docs {rev}", "--parts", part_counts,
                         "--pages-dir", str(build_dir / "pdf_pages"),
                         "--fig-selector", "figure.slate-diagram img"]).returncode
    if rc:
        problems.append(f"{book}: PDF check failed; see " + str(pdf.with_suffix(".check.json")))

    zpath = build_dir / f"{name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(html.rglob("*")):
            if p.is_file():
                z.write(p, f"html/{p.relative_to(html)}")
        if pdf.exists():
            z.write(pdf, pdf.name)
        z.writestr("README.txt", f"{title} (Slate style)\nDocs commit {rev}, built {stamp}\n\n"
                   "Open html/index.html, or read the PDF.\n")
    (build_dir / "placements.json").write_text(json.dumps(placements, indent=1))
    print(f"{book} html:", html / "index.html")
    print(f"{book} pdf:", pdf)
    print(f"{book} zip:", zpath, zpath.stat().st_size, "bytes")
    print(f"{book} problems: {len(problems)}")
    for p in problems:
        print("  -", p)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", choices=["design", "walkthrough", "both"], default="both")
    books = ["design", "walkthrough"] if ap.parse_args().book == "both" else [ap.parse_args().book]
    return max(build(b) for b in books)


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
html_css_files = ["mseg_slate.css"]
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
