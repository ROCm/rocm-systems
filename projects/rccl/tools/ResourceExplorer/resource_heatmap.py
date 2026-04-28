#!/usr/bin/env python3
"""Render a per-column colormap heatmap of *.resources.json fields to a PDF and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def _cmap(name: str) -> matplotlib.colors.Colormap:
    r = getattr(matplotlib, "colormaps", None)
    if r is not None:
        return r[name]  # matplotlib >= 3.5 registry
    import matplotlib.cm as cm
    return cm.get_cmap(name)  # type: ignore[return-value, no-any-return]

_COLUMN_CMAPS = (
    "viridis", "plasma", "cividis", "inferno", "magma", "turbo",
    "Greens", "YlOrRd", "PuBu", "Oranges", "RdPu", "GnBu", "YlGnBu",
    "Reds", "Blues", "BrBG", "PRGn",
)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _format_cell_value(val: Any, max_len: int = 14) -> str:
    if val is None:
        return "—"
    if _is_num(val):
        f = float(val)
        if not np.isfinite(f):
            return "?"
        if abs(f - round(f)) < 1e-6 and abs(f) < 1e12:
            return str(int(round(f)))
        t = f"{f:.4g}"
        if len(t) > max_len:
            return t[: max_len - 1] + "…"
        return t
    if isinstance(val, (dict, list)):
        j = json.dumps(val, sort_keys=True, separators=(",", ":"))
        if len(j) > max_len:
            return j[: max_len - 1] + "…"
        return j
    s = str(val)
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _text_color_f(rgb: np.ndarray) -> str:
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if np.isnan(lum) or lum + r + g + b >= 2.95:  # empty / default white
        return "#202020"
    return "black" if lum > 0.62 else "white"


def _load_dir(json_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    if not json_dir.is_dir():
        print(f"Not a directory: {json_dir}", file=sys.stderr)
        sys.exit(1)
    rows: list[tuple[str, dict[str, Any]]] = []
    for p in sorted(json_dir.glob("*.json")):
        with p.open(encoding="utf-8") as f:
            rows.append((p.name, json.load(f)))
    if not rows:
        print(f"No JSON files in {json_dir}", file=sys.stderr)
        sys.exit(1)
    return rows


def _union_keys(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    s: set[str] = set()
    for _, d in rows:
        s |= d.keys()
    return sorted(s)


def _column_array(dlist: list[dict[str, Any]], col: str) -> np.ndarray:
    n = len(dlist)
    out = np.full(n, np.nan, dtype=np.float64)
    raw = [d.get(col) for d in dlist]
    if all((x is None or _is_num(x)) for x in raw):
        for i, x in enumerate(raw):
            if x is not None:
                out[i] = float(x)
        return out
    levels: dict[str, int] = {}
    for i, x in enumerate(raw):
        if x is None:
            continue
        t = str(x) if not isinstance(x, (dict, list)) else json.dumps(x, sort_keys=True)
        if t not in levels:
            levels[t] = len(levels)
        out[i] = float(levels[t])
    return out


def _norm_column(z: np.ndarray) -> np.ndarray:
    t = z.astype(np.float64, copy=True)
    m = ~np.isnan(t)
    if m.sum() == 0:
        return t
    lo, hi = float(t[m].min()), float(t[m].max())
    if hi - lo < 1e-9:
        t[m] = 0.0
    else:
        t[m] = (t[m] - lo) / (hi - lo)
    t[~m] = np.nan
    return t


def _build_rgba(data: np.ndarray) -> np.ndarray:
    n_r, n_c = data.shape
    out = np.ones((n_r, n_c, 3), dtype=np.float32)
    for j in range(n_c):
        t = _norm_column(data[:, j].copy())
        cfunc = _cmap(_COLUMN_CMAPS[j % len(_COLUMN_CMAPS)])
        mask = np.isnan(t) | np.isnan(data[:, j])
        tc = t.astype(np.float64, copy=True)
        np.nan_to_num(tc, copy=False, nan=0.0)
        np.clip(tc, 0, 1, out=tc)
        rgba = np.asarray(cfunc(tc), dtype=np.float32)
        out[:, j, :3] = rgba[:, :3]
        out[mask, j, :] = 1.0
    return out


def _csv_value(val: Any) -> str:
    """String for CSV cells (round-trip friendly; no PDF truncation)."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if _is_num(val):
        f = float(val)
        if not np.isfinite(f):
            return "nan" if np.isnan(f) else "inf" if f > 0 else "-inf"
        if abs(f - round(f)) < 1e-9 and abs(f) < 1e12:
            return str(int(round(f)))
        return str(f)
    if isinstance(val, (dict, list)):
        return json.dumps(val, sort_keys=True, separators=(",", ":"))
    return str(val)


def _export_csv(
    out_path: Path,
    names: list[str],
    keys: list[str],
    dlist: list[dict[str, Any]],
) -> None:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["file", *keys])
        for name, d in zip(names, dlist):
            w.writerow([name] + [_csv_value(d.get(k)) for k in keys])


def _shorten_filename(name: str, head: int = 28, tail: int = 20) -> str:
    n = len(name)
    if n <= head + tail + 1:
        return name
    return f"{name[:head]}…{name[-tail:]}"


def _render_pdf_one_page(
    row_labels: list[str],
    col_names: list[str],
    rgb_chunk: np.ndarray,
    cell_text: list[list[str]],
    y_stride: int,
    page_title: str,
) -> Figure:
    """Build one heatmap page; caller saves the returned figure to PDF (single or multi-page)."""
    n_r, n_c = rgb_chunk.shape[:2]
    y_stride = max(1, y_stride)
    in_per_col = 0.48
    in_per_row = 0.22
    w_in = max(10.0, n_c * in_per_col + 2.0)
    h_in = max(4.0, n_r * in_per_row + 2.9)
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=200)
    ax.imshow(
        rgb_chunk,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(-0.5, n_c - 0.5, n_r - 0.5, -0.5),
        rasterized=True,
    )
    # Fixed pt size: matches physical cell size when each page is ~letter height.
    fs_cell = min(7.0, max(3.2, 0.36 * in_per_row * 72.0))
    for i in range(n_r):
        for j in range(n_c):
            lab = cell_text[i][j]
            if not lab:
                continue
            if lab == "—":
                tcol = (
                    "#5a5a5a"
                    if _text_color_f(rgb_chunk[i, j]) == "black"
                    else "#c8c8c8"
                )
            else:
                tcol = _text_color_f(rgb_chunk[i, j])
            ax.text(
                float(j),
                float(i),
                lab,
                ha="center",
                va="center",
                color=tcol,
                fontsize=fs_cell,
                family="monospace",
                clip_on=True,
            )
    ax.set_xticks(np.arange(n_c + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_r + 1) - 0.5, minor=True)
    ax.grid(
        which="minor",
        color="white" if n_r * n_c < 5000 else "#f0f0f0",
        linestyle="-",
        linewidth=0.4,
    )
    ax.tick_params(which="minor", size=0)
    n_y = 1 + (n_r - 1) // y_stride
    fsy = min(2.0, max(0.3, 90.0 / max(1, n_y) ** 0.5))
    ypos = np.arange(0, n_r, y_stride, dtype=int)
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [_shorten_filename(row_labels[i]) for i in ypos],
        fontsize=fsy,
        family="monospace",
    )
    x_stride = 1
    if n_c > 48:
        x_stride = max(1, n_c // 40)
    x_show = list(range(0, n_c, x_stride))
    ax.set_xticks(x_show)
    xlabs: list[str] = []
    for j in x_show:
        c = col_names[j]
        if len(c) <= 20:
            xlabs.append(c.replace(" ", "\n"))
        elif len(c) <= 28:
            xlabs.append(c[:6] + "…\n" + c[-4:])
        else:
            xlabs.append(c[:6] + "…" + c[-3:])
    x_fs = max(5, min(8, 200 / n_c**0.25 if n_c else 7))
    ax.set_xticklabels(
        xlabs, rotation=55, ha="right", fontsize=x_fs, fontweight="light"
    )
    ax.set_xlabel("Field (separate colormap and scale per column)", fontsize=9)
    ax.set_ylabel("Resource JSON file", fontsize=9)
    ax.set_title(page_title, fontsize=10, pad=6)
    for _, spine in ax.spines.items():
        spine.set_visible(False)
    fig.tight_layout()
    fig.patch.set_facecolor("white")
    return fig


def _render_pdf(
    row_labels: list[str],
    col_names: list[str],
    data: np.ndarray,
    cell_text: list[list[str]],
    out_path: Path,
    y_stride: int,
    rows_per_page: int,
) -> int:
    n_r, n_c = data.shape
    y_stride = max(1, y_stride)
    rgb = _build_rgba(data)
    in_per_row = 0.22
    if rows_per_page <= 0:
        rows_per_page = n_r
    else:
        # Keep one page to a viewable height (~9–10 in data) so the viewer does
        # not scale thousands of rows onto one page.
        max_sane = max(1, int(8.0 / in_per_row))
        rows_per_page = max(1, min(rows_per_page, max_sane))
    n_pages = max(1, (n_r + rows_per_page - 1) // rows_per_page)
    if n_pages == 1:
        title = "RCCL device resource fields"
        if len(row_labels) > 0 and n_r > 1:
            title = f"RCCL device resource fields (rows 1–{n_r} of {n_r})"
        fig = _render_pdf_one_page(
            row_labels, col_names, rgb, cell_text, y_stride, title
        )
        fig.savefig(
            out_path,
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
            dpi=200,
        )
        plt.close(fig)
        return 1
    with PdfPages(out_path) as pdf:
        for p in range(n_pages):
            lo = p * rows_per_page
            hi = min(n_r, lo + rows_per_page)
            rchunk = rgb[lo:hi, :, :]
            labels = row_labels[lo:hi]
            tchunk = [cell_text[i] for i in range(lo, hi)]
            title = (
                f"RCCL device resource fields (rows {lo + 1}–{hi} of {n_r}, "
                f"page {p + 1}/{n_pages})"
            )
            fig = _render_pdf_one_page(
                labels, col_names, rchunk, tchunk, y_stride, title
            )
            pdf.savefig(fig, bbox_inches="tight", facecolor="white", dpi=200)
            plt.close(fig)
    return n_pages


def main() -> None:
    default_in = (
        Path.home() / "rccl/build/release/src/CMakeFiles/rccl_device_gfx950.dir"
        "/__/hipify/gensrc/specialized"
    )
    ap = argparse.ArgumentParser(
        description="Build PDF heatmap of JSON resource fields (rows=files, cols=fields).",
    )
    ap.add_argument(
        "json_dir",
        nargs="?",
        type=Path,
        default=default_in,
        help=f"Directory of *.json (default: {default_in})",
    )
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: resource_fields_heatmap.pdf next to this script).",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output CSV path (default: same basename as --output with .csv).",
    )
    ap.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write a CSV file.",
    )
    ap.add_argument(
        "--y-stride",
        type=int,
        default=1,
        metavar="N",
        help="Y-axis label step (1 = all filenames; use >1 for very large sets / faster export).",
    )
    ap.add_argument(
        "--rows-per-page",
        type=int,
        default=32,
        metavar="N",
        help="Split output into a multi-page PDF: at most this many data rows per page (default: 32). "
        "Capped to keep each page a readable on-screen size. Use 0 for a single (often huge) page.",
    )
    args = ap.parse_args()
    in_dir = args.json_dir.expanduser()
    out = args.output
    if out is None:
        out = Path(__file__).resolve().parent / "resource_fields_heatmap.pdf"

    entries = _load_dir(in_dir)
    keys = _union_keys(entries)
    dlist = [d for _, d in entries]
    names = [n for n, _ in entries]
    arr = np.full((len(entries), len(keys)), np.nan, dtype=np.float64)
    for j, k in enumerate(keys):
        arr[:, j] = _column_array(dlist, k)
    cell_text = [
        [_format_cell_value(d.get(k)) for k in keys] for d in dlist
    ]
    csv_path: Path | None = None
    if not args.no_csv:
        csv_path = args.csv if args.csv is not None else out.with_suffix(".csv")
        _export_csv(csv_path, names, keys, dlist)

    n_pages = _render_pdf(
        names, keys, arr, cell_text, out, args.y_stride, args.rows_per_page
    )
    line = (
        f"Wrote {out} ({len(names)} x {len(keys)}), {n_pages} page(s), "
        f"y-stride {args.y_stride}"
    )
    if csv_path is not None:
        line += f"; CSV: {csv_path}"
    print(line)


if __name__ == "__main__":
    main()
