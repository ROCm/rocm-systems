#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Replace AMD copyright headers with short-form SPDX header.

Usage:
    python3 tools/update_copyright.py [--dryrun] [REPO_ROOT]

If REPO_ROOT is not specified, the parent directory of this script is used.
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# New header templates
# ---------------------------------------------------------------------------
HASH_HEADER = (
    "# Copyright (c) Advanced Micro Devices, Inc.\n# SPDX-License-Identifier:  MIT\n"
)
SLASH_HEADER = (
    "// Copyright (c) Advanced Micro Devices, Inc.\n// SPDX-License-Identifier:  MIT\n"
)

# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------
HASH_EXTENSIONS = {".py", ".sh", ".cmake"}
SLASH_EXTENSIONS = {".cpp", ".hpp", ".h", ".hip", ".c", ".cc", ".cxx"}

# Extensionless files that use # comment style
KNOWN_HASH_FILES = {"rocprof-compute"}


def get_comment_style(path: Path):
    """Return '#' or '//' based on file extension/name, or None to skip."""
    ext = path.suffix.lower()
    name = path.name

    if ext in HASH_EXTENSIONS:
        return "#"
    if ext in SLASH_EXTENSIONS:
        return "//"
    if name.startswith("Dockerfile"):
        return "#"
    if name in KNOWN_HASH_FILES:
        return "#"
    if ext == "":
        # Extensionless: check for shebang
        try:
            first = path.read_bytes()[:64].decode("utf-8", errors="replace")
            if first.startswith("#!"):
                return "#"
        except OSError:
            pass
    return None


# ---------------------------------------------------------------------------
# Exclusion rules
# ---------------------------------------------------------------------------
EXCLUDED_DIR_PARTS = {
    "pyyaml",  # src/vendored/pyyaml
    "fontello",  # assets/fontello
    ".git",
    "__pycache__",
    "build",
    ".rocprofv3",
}

# Additional path-substring exclusions (checked against the path relative to repo root)
EXCLUDED_PATH_SUBSTRINGS = [
    "src/rocprof_compute_analyze/assets/fonts",
    "docs/archive",
    "DEBIAN",
]

EXCLUDED_FILENAMES = {"LICENSE.md"}


def is_excluded(path: Path, repo_root: Path) -> bool:
    rel = path.relative_to(repo_root)
    # Check each path part
    for part in rel.parts:
        if part in EXCLUDED_DIR_PARTS:
            return True
    # Check substring rules against the posix relative path
    rel_str = rel.as_posix()
    for sub in EXCLUDED_PATH_SUBSTRINGS:
        if rel_str.startswith(sub):
            return True
    if path.name in EXCLUDED_FILENAMES:
        return True
    return False


# ---------------------------------------------------------------------------
# AMD copyright marker — skip files that don't have one
# ---------------------------------------------------------------------------
AMD_MARKER = "Advanced Micro Devices"


# ---------------------------------------------------------------------------
# Header detection and replacement (line-by-line parser)
# ---------------------------------------------------------------------------

RE_HASH_SEP = re.compile(r"^#{10,}\s*$")  # plain ####...####
RE_HASH_BL = re.compile(r"^#{10,}bl\s*$")  # ####...####bl
RE_HASH_EL = re.compile(r"^#{10,}el\s*$")  # ####...####el
RE_SLASH_MIT = re.compile(r"^// MIT License\s*$")  # // MIT License
RE_SLASH_LINE = re.compile(r"^//")  # any // line


def _is_preamble_line(line: str) -> bool:
    """Lines that come before the copyright block and must be preserved."""
    stripped = line.strip()
    if stripped.startswith("#!"):  # shebang
        return True
    if stripped.startswith('"""') or stripped.startswith("'''"):  # docstring
        return True
    if stripped == "":  # blank line between shebang and header
        return True
    if stripped.startswith("# ruff:") or stripped.startswith("# type:"):
        return True
    return False


def process_content(lines: list, comment_style: str) -> tuple:
    """
    Detect and remove the AMD copyright header from *lines*.

    Returns (preamble, header_end_idx, descriptive_comment) where:
      - preamble: lines before the copyright block (preserved as-is)
      - header_end_idx: index in *lines* of the first line after the old header
      - descriptive_comment: text to re-wrap in /* ... */ for Format D files, or None

    Returns None if no AMD copyright header is detected.
    """
    n = len(lines)

    # ------------------------------------------------------------------
    # Format D/E: /* ... */ wrapping # MIT License block (C/C++/HIP files)
    # These start at line 0 with "/*"
    # ------------------------------------------------------------------
    if lines and lines[0].rstrip() == "/*":
        # Scan the entire /* ... */ block
        end_idx = None
        for i in range(1, n):
            if lines[i].rstrip() == "*/":
                end_idx = i
                break
        if end_idx is not None:
            block = "".join(lines[1:end_idx])
            if "MIT License" in block and AMD_MARKER in block:
                # Found Format D or E.  Extract descriptive text that follows
                # the second ####...#### or ####el line within the block.
                inner = lines[1:end_idx]
                desc_start = None
                # Find the end-of-license separator inside the block
                for j, ln in enumerate(inner):
                    if RE_HASH_EL.match(ln) or (RE_HASH_SEP.match(ln) and j > 0):
                        desc_start = j + 1
                        break
                descriptive = None
                if desc_start is not None:
                    desc_lines = inner[desc_start:]
                    # Strip leading/trailing blank lines
                    desc_text = "".join(desc_lines).strip()
                    if desc_text:
                        descriptive = desc_text
                return [], end_idx + 1, descriptive

    # ------------------------------------------------------------------
    # Identify preamble (lines before the copyright block)
    # Preamble = shebang + optional blank + optional docstring + optional blank
    # We stop at the first line that looks like a copyright block start.
    # ------------------------------------------------------------------
    preamble_end = 0
    i = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()
        # Stop at copyright block starts
        if RE_HASH_BL.match(stripped):
            break
        if RE_HASH_SEP.match(stripped):
            # Only preamble if next non-empty line is "# MIT License"
            for j in range(i + 1, min(i + 3, n)):
                if lines[j].strip():
                    if lines[j].strip() == "# MIT License":
                        # This is Format B start
                        break
                    else:
                        # Not a copyright block; treat as preamble
                        i += 1
                        preamble_end = i
                        break
            else:
                break
            break
        if RE_SLASH_MIT.match(stripped):
            break
        if line.rstrip() == "/*":
            break  # already handled above
        # Preamble lines: shebang, docstrings, linter directives
        if (
            stripped.startswith("#!")
            or stripped.startswith('"""')
            or stripped.startswith("'''")
            or stripped.startswith("# ruff:")
            or stripped.startswith("# type:")
            or stripped.startswith("# noqa")
            or stripped.startswith("# fmt:")
        ):
            preamble_end = i + 1
            i += 1
            continue
        if stripped == "":
            # A blank line in the preamble region is kept only if
            # we've seen a real preamble line
            if preamble_end > 0:
                preamble_end = i + 1
            i += 1
            continue
        # Any other line before a copyright block: stop preamble collection
        break

    preamble = lines[:preamble_end]
    i = preamble_end

    # ------------------------------------------------------------------
    # Format A: ####bl ... ####el
    # ------------------------------------------------------------------
    if i < n and RE_HASH_BL.match(lines[i].strip()):
        # Scan forward for ####el
        for j in range(i + 1, min(i + 60, n)):
            if RE_HASH_EL.match(lines[j].strip()):
                return preamble, j + 1, None
        # No el found — not a valid header
        return None

    # ------------------------------------------------------------------
    # Format B: ####  (plain) + # MIT License ... # THE SOFTWARE. + blank + ####
    # ------------------------------------------------------------------
    if i < n and RE_HASH_SEP.match(lines[i].strip()):
        # Confirm next non-empty line is "# MIT License"
        confirmed = False
        for k in range(i + 1, min(i + 3, n)):
            if lines[k].strip():
                if lines[k].strip() == "# MIT License":
                    confirmed = True
                break
        if confirmed:
            # Scan forward to find end: # THE SOFTWARE. or # SOFTWARE.
            # then optional blank line(s) then closing ####
            j = i + 1
            while j < n:
                s = lines[j].strip()
                if s in ("# THE SOFTWARE.", "# SOFTWARE."):
                    j += 1
                    break
                j += 1
            # j now points to the line after "# THE/SOFTWARE." (or ran off end)
            # Consume optional blank lines then closing ####
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n and RE_HASH_SEP.match(lines[j].strip()):
                j += 1  # consume the closing ####
            return preamble, j, None

    # ------------------------------------------------------------------
    # Format C: // MIT License ... // SOFTWARE.
    # ------------------------------------------------------------------
    if i < n and RE_SLASH_MIT.match(lines[i].strip()):
        j = i + 1
        last_slash_line = i
        while j < n:
            if RE_SLASH_LINE.match(lines[j]):
                last_slash_line = j
                j += 1
            else:
                break
        # last_slash_line is the last // line; that is the end of the block
        return preamble, last_slash_line + 1, None

    return None


def process_file(path: Path, dryrun: bool) -> bool:
    """
    Process a single file: detect and replace its AMD copyright header.
    Returns True if the file was changed (or would be changed in dryrun mode).
    """
    try:
        content = path.read_text(encoding="utf-8", errors="surrogateescape")
    except (OSError, UnicodeDecodeError):
        return False

    if AMD_MARKER not in content:
        return False

    comment_style = get_comment_style(path)
    if comment_style is None:
        return False

    lines = content.splitlines(keepends=True)
    result = process_content(lines, comment_style)
    if result is None:
        return False

    preamble, header_end, descriptive = result

    new_header = HASH_HEADER if comment_style == "#" else SLASH_HEADER

    # Build new content
    new_lines = list(preamble)
    new_lines.append(new_header)

    # Add the descriptive comment (Format D/E only)
    if descriptive:
        new_lines.append("\n")
        new_lines.append("/*\n")
        new_lines.append(descriptive + "\n")
        new_lines.append("*/\n")

    # Rest of file after old header
    rest = lines[header_end:]

    # Strip leading blank lines from rest (we'll add exactly one blank line)
    while rest and rest[0].strip() == "":
        rest = rest[1:]

    if rest:
        new_lines.append("\n")
        new_lines.extend(rest)

    new_content = "".join(new_lines)

    if new_content == content:
        return False

    if dryrun:
        print(f"  Would update: {path}")
        return True

    # Write atomically via temp file
    tmp = path.with_name("." + path.name + ".tmp")
    try:
        tmp.write_text(new_content, encoding="utf-8", errors="surrogateescape")
        shutil.copystat(path, tmp)
        tmp.replace(path)
    except Exception as exc:
        print(f"  ERROR writing {path}: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    print(f"  Updated: {path}")
    return True


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------


def walk_repo(repo_root: Path, dryrun: bool) -> int:
    """Walk the repo and process all eligible files. Returns count of changed files."""
    changed = 0
    for dirpath, dirnames, filenames in os.walk(repo_root):
        current = Path(dirpath)

        # Prune excluded directories in-place so os.walk doesn't descend into them
        dirnames[:] = [
            d
            for d in dirnames
            if d not in EXCLUDED_DIR_PARTS and not is_excluded(current / d, repo_root)
        ]

        for fname in filenames:
            fpath = current / fname
            if is_excluded(fpath, repo_root):
                continue
            comment_style = get_comment_style(fpath)
            if comment_style is None:
                continue
            if process_file(fpath, dryrun):
                changed += 1

    return changed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Replace AMD MIT License copyright headers with "
        "short-form SPDX header."
    )
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=None,
        help="Repository root directory (default: parent of this script)",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Preview changes without writing files",
    )
    args = parser.parse_args()

    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = Path(__file__).resolve().parent.parent

    if not repo_root.is_dir():
        print(f"ERROR: {repo_root} is not a directory", file=sys.stderr)
        sys.exit(1)

    mode = "DRY RUN" if args.dryrun else "APPLY"
    print(f"update_copyright.py [{mode}]")
    print(f"Repo root: {repo_root}\n")

    changed = walk_repo(repo_root, args.dryrun)

    print(
        f"\n{'Files that would change' if args.dryrun else 'Files changed'}: {changed}"
    )
    if args.dryrun and changed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
