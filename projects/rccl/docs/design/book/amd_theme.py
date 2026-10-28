"""AMD-style Mermaid theme modeled on the HIP programming-model figures, light for print.

Very faint grey surface (#f7f7f8), white nodes with AMD gray (#585556) outlines,
charcoal (#312f37) Open Sans text and AMD red (#8f1414) for failure outcomes.
The docs' component and outcome classes keep their meaning; only their tints change.
"""
from __future__ import annotations

import re
from pathlib import Path

import mermaid_svg as M
import rocm_docs

FONT_DIR = Path(rocm_docs.__file__).parent / "rocm_docs_theme/static/fonts/open-sans"
FONT_CSS = M.font_face_css("Open Sans", [(400, FONT_DIR / "open-sans-v34-latin-regular.woff2"),
                                         (600, FONT_DIR / "open-sans-v34-latin-600.woff2")])
FONT = "'Open Sans', Arial, sans-serif"

SURFACE = "#f7f7f8"
INK = "#312f37"
# Mermaid sketches flowcharts itself; mermaid_svg.render sketches sequence diagrams with rough.js.
CFG = dict(M.READABLE, theme="base", look="handDrawn", handDrawnSeed=7, themeVariables={
    "darkMode": False, "fontFamily": FONT, "fontSize": "16px", "background": SURFACE,
    "primaryColor": "#ffffff", "primaryBorderColor": "#585556", "primaryTextColor": INK,
    "secondaryColor": "#f1f1f3", "tertiaryColor": "#ececee", "lineColor": "#585556", "textColor": INK,
    "clusterBkg": "#eeeef0", "clusterBorder": "#c9c9cc", "titleColor": INK,
    "edgeLabelBackground": SURFACE, "nodeTextColor": INK,
    "noteBkgColor": "#f8eaea", "noteBorderColor": "#8f1414", "noteTextColor": INK,
    "actorBkg": "#ffffff", "actorBorder": "#585556", "actorTextColor": INK, "actorLineColor": "#9d9fa2",
    "signalColor": "#444246", "signalTextColor": INK, "labelBoxBkgColor": "#f8eaea",
    "labelBoxBorderColor": "#8f1414", "labelTextColor": INK, "loopTextColor": INK,
    "activationBkgColor": "#eeeef0", "activationBorderColor": "#585556", "sequenceNumberColor": "#ffffff",
    # gantt
    "sectionBkgColor": "#efeff1", "altSectionBkgColor": SURFACE, "sectionBkgColor2": "#efeff1",
    "taskBkgColor": "#f3d9d9", "taskBorderColor": "#8f1414", "taskTextColor": INK,
    "taskTextLightColor": INK, "taskTextOutsideColor": INK, "taskTextDarkColor": INK,
    "activeTaskBkgColor": "#ffffff", "activeTaskBorderColor": "#585556", "doneTaskBkgColor": "#e4e4e7",
    "doneTaskBorderColor": "#9d9fa2", "gridColor": "#d6d6da", "todayLineColor": "#8f1414",
})

# Existing book fill, stroke -> AMD book fill, stroke, per legend class: soft tints, muted strokes.
CLASS_MAP = {
    ("#e5f1ff", "#007aff"): ("#e8f0fa", "#2f6db5"),  # gin
    ("#f5ebfb", "#af52de"): ("#f2ecf8", "#7b4fa6"),  # rma
    ("#e0f5f8", "#008299"): ("#e6f4f5", "#0f7c86"),  # netib
    ("#fff1e0", "#ff9500"): ("#fcf1e4", "#c26a10"),  # ibcast
    ("#f2f2f7", "#8e8e93"): ("#efeff1", "#77767b"),  # hostmem
    ("#eeeefb", "#5856d6"): ("#ededf8", "#4f4cb0"),  # devmem
    ("#e8f8ec", "#34c759"): ("#e9f5ec", "#2e8b46"),  # okpath
    ("#fff9e0", "#ffcc00"): ("#fbf5de", "#a8860a"),  # stagepath
    ("#ffeceb", "#ff3b30"): ("#f8e4e4", "#8f1414"),  # failpath: AMD red
    ("#f9f9fb", "#d1d1d6"): ("#ffffff", "#b5b5ba"),  # neutral
}


def _rgb(h: str) -> str:
    return f"rgb({int(h[1:3], 16)},{int(h[3:5], 16)},{int(h[5:7], 16)})"


def recolor(src: str) -> str:
    """Rewrite classDef colors and sequence box/rect tints to the AMD palette."""
    out = src
    for (lf, ls), (df, ds) in CLASS_MAP.items():
        out = re.sub(rf"fill:{lf}\s*,\s*stroke:{ls}", f"fill:{df},stroke:{ds}", out, flags=re.I)
        out = re.sub(re.escape(_rgb(lf)).replace(",", r",\s*"), _rgb(df), out)
    return re.sub(r"color:#1d1d1f", f"color:{INK}", out, flags=re.I)
