###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""
Source-location correlation helper for PerfXpert formatters.

Cross-references Tier-1 hotspot kernel names with Tier-0 ``detected_kernels``
entries so each hotspot row can carry an optional ``source_locations`` list.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


_LAUNCH_KIND = {
    "GLOBAL_KERNEL_DEF": "definition",
    "HIP_KERNEL_LAUNCH": "launch",
    "TRIPLE_ANGLE_LAUNCH": "launch",
}

_SEVERITY_HIGH = ("HIGH", "CRITICAL", "#e84040")
_SEVERITY_MEDIUM = ("MEDIUM", "HOT", "#f08432")
_SEVERITY_LOW = ("LOW", "WARM", "#caa828")
_SEVERITY_INFO = ("INFO", "COOL", "#4d8ef2")


def _classify_severity(percent_of_total: float) -> Tuple[str, str, str]:
    """Bucket ``percent_of_total`` into one of 4 severities."""
    try:
        p = float(percent_of_total or 0.0)
    except (TypeError, ValueError):
        p = 0.0
    if p >= 20.0:
        return _SEVERITY_HIGH
    if p >= 5.0:
        return _SEVERITY_MEDIUM
    if p >= 1.0:
        return _SEVERITY_LOW
    return _SEVERITY_INFO


def _demangle_basename(name: str) -> str:
    """Canonicalize a kernel name for comparison."""
    if not name:
        return ""
    s = str(name)
    paren = s.find("(")
    if paren != -1:
        s = s[:paren]
    out = []
    depth = 0
    for ch in s:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            if depth > 0:
                depth -= 1
            continue
        if depth == 0:
            out.append(ch)
    s = "".join(out)
    if "::" in s:
        s = s.rsplit("::", 1)[-1]
    s = s.strip().strip("*&")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _kernel_entry_to_location(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a Tier-0 ``detected_kernels`` entry to a ``source_locations`` item."""
    file_ = entry.get("file")
    line = entry.get("line")
    launch_type = entry.get("launch_type") or "GLOBAL_KERNEL_DEF"
    if not file_ or line is None:
        return None
    kind = _LAUNCH_KIND.get(launch_type, "definition")
    return {
        "file": str(file_),
        "line": int(line),
        "kind": kind,
        "launch_type": launch_type,
    }


def correlate_hotspots_with_source(
    hotspots: Iterable[Dict[str, Any]],
    detected_kernels: Optional[Iterable[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Annotate each hotspot with a ``source_locations`` list."""
    index: Dict[str, List[Dict[str, Any]]] = {}
    for k in detected_kernels or []:
        name = k.get("name")
        loc = _kernel_entry_to_location(k)
        if not name or loc is None:
            continue
        key = _demangle_basename(name)
        if not key:
            continue
        index.setdefault(key, []).append(loc)

    annotated: List[Dict[str, Any]] = []
    for h in hotspots or []:
        out = dict(h)
        hname = out.get("name") or ""
        key = _demangle_basename(hname)
        locs = index.get(key, []) if key else []
        sev_id, sev_label, sev_color = _classify_severity(
            out.get("percent_of_total", 0.0)
        )
        kind_rank = {"definition": 0, "launch": 1}
        locs = sorted(
            (dict(loc) for loc in locs),
            key=lambda lo: (
                kind_rank.get(lo.get("kind"), 9),
                lo.get("file", ""),
                lo.get("line", 0),
            ),
        )
        for lo in locs:
            lo["severity"] = sev_id
            lo["severity_label"] = sev_label
            lo["severity_color"] = sev_color
        out["source_locations"] = locs
        annotated.append(out)
    return annotated


def format_source_citation_inline(locations: Optional[List[Dict[str, Any]]]) -> str:
    """Render ``file.hip:42 (definition), file.hip:58 (launch)``."""
    if not locations:
        return ""
    parts = []
    for lo in locations:
        f = lo.get("file", "?")
        line = lo.get("line", "?")
        kind = lo.get("kind", "definition")
        parts.append(f"{f}:{line} ({kind})")
    rendered = ", ".join(parts)
    label = None
    for lo in locations:
        label = lo.get("severity_label")
        if label:
            break
    if label:
        rendered = f"{rendered} [{label}]"
    return rendered


__all__ = [
    "correlate_hotspots_with_source",
    "format_source_citation_inline",
    "_classify_severity",
    "_demangle_basename",
]
