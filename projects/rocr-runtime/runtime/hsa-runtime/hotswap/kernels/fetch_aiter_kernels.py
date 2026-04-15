#!/usr/bin/env python3
"""Download precompiled AITER CK kernels (.co) from github.com/ROCm/aiter.

These are the same code objects the original hotswap was developed against
(1,315 kernels across 12 categories for gfx950).

Usage:
    python3 fetch_aiter_kernels.py                  # representative subset (~30 kernels)
    python3 fetch_aiter_kernels.py --full            # all kernels (~500+ from git repo)
    python3 fetch_aiter_kernels.py --arch gfx942     # different architecture
    python3 fetch_aiter_kernels.py --list             # just list available files
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

REPO = "ROCm/aiter"
BRANCH = "main"
API_BASE = f"https://api.github.com/repos/{REPO}/contents"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

# Representative subset: 1-3 files per category to cover the breadth of
# kernel types without downloading hundreds of files.
REPRESENTATIVE_SUBSET = {
    "bf16gemm": 2,
    "f4gemm": 2,
    "fmha_v3_fwd": 3,
    "fmha_v3_bwd": 2,
    "fmoe": 2,
    "fmoe_2stages": 2,
    "fp8gemm_blockscale": 2,
    "i8gemm": 2,
    "mla": 2,
    "pa": 2,
    "topksoftmax": 2,
    # Loose .co files at the top level (f8_block_scale variants)
    ".": 99,
}


def api_get(path: str) -> list[dict]:
    url = f"{API_BASE}/{path}?ref={BRANCH}"
    req = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"ERROR: GitHub API rate limit exceeded. Set GITHUB_TOKEN env var.", file=sys.stderr)
            print(f"  export GITHUB_TOKEN=$(gh auth token)", file=sys.stderr)
            sys.exit(1)
        raise


def discover_co_files(base_path: str, recurse: bool = True) -> list[dict]:
    """Recursively discover all .co files under a GitHub directory."""
    entries = api_get(base_path)
    co_files = []
    for entry in entries:
        if entry["type"] == "file" and entry["name"].endswith(".co"):
            co_files.append(entry)
        elif entry["type"] == "dir" and recurse:
            co_files.extend(discover_co_files(entry["path"], recurse=True))
    return co_files


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"token {token}")
    with urllib.request.urlopen(req) as resp:
        dest.write_bytes(resp.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="Download ALL .co files (500+), not just the representative subset")
    parser.add_argument("--arch", default="gfx950",
                        help="GPU architecture to download (default: gfx950)")
    parser.add_argument("--list", action="store_true",
                        help="List available files without downloading")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: <script_dir>/aiter_<arch>)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    out_dir = args.output or (script_dir / f"aiter_{args.arch}")
    hsa_path = f"hsa/{args.arch}"

    print(f"Discovering .co files in {REPO}/{hsa_path} ...")

    # Discover top-level structure
    top_entries = api_get(hsa_path)
    dirs = [e for e in top_entries if e["type"] == "dir"]
    loose_co = [e for e in top_entries if e["type"] == "file" and e["name"].endswith(".co")]

    all_files: dict[str, list[dict]] = {}
    if loose_co:
        all_files["."] = loose_co

    for d in dirs:
        category = d["name"]
        print(f"  Scanning {category}/ ...", end="", flush=True)
        co_files = discover_co_files(d["path"])
        all_files[category] = co_files
        print(f" {len(co_files)} .co files")

    total = sum(len(v) for v in all_files.values())
    print(f"\nTotal: {total} .co files across {len(all_files)} categories")

    if args.list:
        for category, files in sorted(all_files.items()):
            print(f"\n=== {category} ({len(files)} files) ===")
            for f in files:
                print(f"  {f['name']}  ({f['size']} bytes)")
        return

    # Select files to download
    to_download: list[dict] = []
    if args.full:
        for files in all_files.values():
            to_download.extend(files)
    else:
        for category, files in all_files.items():
            limit = REPRESENTATIVE_SUBSET.get(category, 2)
            selected = files[:limit]
            to_download.extend(selected)
            if len(files) > limit:
                print(f"  {category}: selected {limit}/{len(files)} (use --full for all)")

    print(f"\nDownloading {len(to_download)} .co files to {out_dir}/ ...")

    for i, entry in enumerate(to_download, 1):
        # Preserve directory structure relative to hsa/<arch>/
        rel_path = entry["path"].removeprefix(f"hsa/{args.arch}/")
        dest = out_dir / rel_path
        size_kb = entry["size"] / 1024
        print(f"  [{i}/{len(to_download)}] {rel_path} ({size_kb:.0f} KB)", flush=True)
        download_file(entry["download_url"], dest)

    print(f"\nDone. {len(to_download)} kernels saved to {out_dir}/")


if __name__ == "__main__":
    main()
