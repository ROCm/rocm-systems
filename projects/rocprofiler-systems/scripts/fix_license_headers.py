#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Replace long-form MIT license headers with SPDX format and add where missing.

Processes .cpp, .hpp, .py, .cmake, and CMakeLists.txt files.
Skips build/, external/, and .git/ directories.
"""

import argparse
import os
import re
import sys

SPDX_COPYRIGHT = "Copyright (c) Advanced Micro Devices, Inc."
SPDX_LICENSE = "SPDX-License-Identifier: MIT"

DEFAULT_EXCLUDED_DIRS = {"build", "external", "examples", ".git"}

# Patterns to match the long-form MIT license block.
# The block starts with "MIT License" and ends with a line containing "SOFTWARE."
MIT_LONG_FORM_MARKERS = [
    "MIT License",
    "Permission is hereby granted, free of charge",
    "THE SOFTWARE IS PROVIDED",
]


def make_spdx_header(comment_prefix):
    """Return the 2-line SPDX header with the given comment prefix."""
    return f"{comment_prefix} {SPDX_COPYRIGHT}\n" f"{comment_prefix} {SPDX_LICENSE}\n"


def get_comment_prefix(filepath):
    """Return the comment prefix for the file type."""
    if filepath.endswith((".cpp", ".hpp")):
        return "//"
    if (
        filepath.endswith((".py", ".cmake"))
        or os.path.basename(filepath) == "CMakeLists.txt"
    ):
        return "#"
    return None


def strip_comment(line, prefix):
    """Strip comment prefix from a line and return the text content."""
    stripped = line.rstrip("\n")
    if stripped.startswith(prefix):
        return stripped[len(prefix) :]
    return None


def find_long_mit_block(lines, prefix):
    """Find the start and end indices of the long-form MIT license block.

    Returns (start, end) where end is exclusive, or None if not found.
    The search looks in the first 30 lines of the file.
    """
    search_limit = min(len(lines), 30)

    # Find the line containing "MIT License"
    mit_start = None
    for i in range(search_limit):
        text = strip_comment(lines[i], prefix)
        if text is not None and "MIT License" in text:
            mit_start = i
            break

    if mit_start is None:
        return None

    # Find the end: line containing "SOFTWARE." (the last line of the MIT text)
    mit_end = None
    for i in range(mit_start, min(len(lines), mit_start + 25)):
        text = strip_comment(lines[i], prefix)
        if text is not None and "SOFTWARE." in text:
            mit_end = i + 1  # exclusive
            break

    if mit_end is None:
        return None

    # Verify it's actually the MIT license by checking for key phrases
    block_text = "\n".join(lines[mit_start:mit_end])
    if not any(marker in block_text for marker in MIT_LONG_FORM_MARKERS[1:]):
        return None

    # Expand to include surrounding empty comment lines (e.g. bare "//" or "#")
    # Look backwards from mit_start
    while mit_start > 0:
        prev = lines[mit_start - 1].rstrip("\n")
        if prev == prefix or prev == f"{prefix} ":
            mit_start -= 1
        else:
            break

    # Look forwards from mit_end
    while mit_end < len(lines):
        nxt = lines[mit_end].rstrip("\n")
        if nxt == prefix or nxt == f"{prefix} ":
            mit_end += 1
        else:
            break

    # Also consume a blank line after the block
    if mit_end < len(lines) and lines[mit_end].strip() == "":
        mit_end += 1

    return (mit_start, mit_end)


def has_spdx_header(lines, limit=5):
    """Check if the file already has an SPDX header in the first few lines."""
    for line in lines[:limit]:
        if "SPDX-License-Identifier:" in line:
            return True
    return False


def has_copyright_line(lines, limit=5):
    """Check if the file has a copyright line in the first few lines."""
    for line in lines[:limit]:
        if "Copyright" in line and "Advanced Micro Devices" in line:
            return True
    return False


def normalize_spdx_spacing(lines, prefix):
    """Normalize SPDX-License-Identifier spacing to single space after colon.

    Returns (modified_lines, changed).
    """
    changed = False
    result = []
    for line in lines:
        # Match SPDX line with any spacing after colon
        new_line = re.sub(
            rf"^(\s*{re.escape(prefix)}\s*)SPDX-License-Identifier:\s+MIT",
            rf"\1SPDX-License-Identifier: MIT",
            line,
        )
        if new_line != line:
            changed = True
        result.append(new_line)
    return result, changed


def process_file(filepath, dry_run=False, verbose=False):
    """Process a single file. Returns action taken as a string, or None."""
    prefix = get_comment_prefix(filepath)
    if prefix is None:
        return None

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if not lines:
        return None

    # Check for shebang
    has_shebang = lines[0].startswith("#!")
    shebang_line = lines[0] if has_shebang else None

    # Search range for headers (skip shebang if present)
    search_start = 1 if has_shebang else 0
    search_lines = lines[search_start:]

    # Case 1: Already has SPDX header — just normalize spacing
    if has_spdx_header(search_lines):
        new_lines, changed = normalize_spdx_spacing(lines, prefix)
        if changed:
            if not dry_run:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
            return "normalized_spacing"
        return "already_spdx"

    # Case 2: Has long-form MIT header — replace with SPDX
    mit_block = find_long_mit_block(lines, prefix)
    if mit_block is not None:
        start, end = mit_block
        spdx = make_spdx_header(prefix)

        # Check if there's a Copyright line with a year range before the MIT block
        # that we also want to include in the removed section
        # (the Copyright line is part of the long-form block)

        new_lines = lines[:start] + [spdx, "\n"] + lines[end:]

        # Clean up double blank lines that might result
        cleaned = []
        prev_blank = False
        for line in new_lines:
            is_blank = line.strip() == ""
            if is_blank and prev_blank:
                continue
            cleaned.append(line)
            prev_blank = is_blank

        if not dry_run:
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(cleaned)
        return "replaced_long_form"

    # Case 3: No license header at all — add SPDX
    spdx = make_spdx_header(prefix)

    if has_shebang:
        new_lines = [shebang_line, spdx, "\n"] + lines[1:]
    else:
        new_lines = [spdx, "\n"] + lines

    # Clean up double blank lines at the insertion point
    cleaned = []
    prev_blank = False
    for line in new_lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    if not dry_run:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(cleaned)
    return "added_header"


def collect_files(root, excluded_dirs):
    """Walk the tree and yield file paths to process."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]

        for filename in sorted(filenames):
            if (
                filename.endswith((".cpp", ".hpp", ".py", ".cmake"))
                or filename == "CMakeLists.txt"
            ):
                yield os.path.join(dirpath, filename)


def main():
    parser = argparse.ArgumentParser(
        description="Replace long-form MIT license with SPDX format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each file and action taken",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Directories to exclude (default: build external examples .git)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Individual files to process, or omit to scan project root",
    )
    args = parser.parse_args()

    excluded_dirs = (
        set(args.exclude) if args.exclude is not None else DEFAULT_EXCLUDED_DIRS
    )

    stats = {
        "replaced_long_form": 0,
        "added_header": 0,
        "normalized_spacing": 0,
        "already_spdx": 0,
        "skipped": 0,
    }

    if args.files:
        # Process individual files
        file_list = args.files
        root = os.getcwd()
    else:
        # Scan project root
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_list = list(collect_files(root, excluded_dirs))
        print(f"  Excluding dirs: {', '.join(sorted(excluded_dirs))}")
        print()

    for filepath in file_list:
        rel = os.path.relpath(filepath, root)
        try:
            action = process_file(filepath, dry_run=args.dry_run, verbose=args.verbose)
        except Exception as e:
            print(f"  ERROR {rel}: {e}", file=sys.stderr)
            stats["skipped"] += 1
            continue

        if action is None:
            stats["skipped"] += 1
        else:
            stats[action] += 1
            if args.verbose or action not in ("already_spdx",):
                label = {
                    "replaced_long_form": "REPLACE",
                    "added_header": "ADD",
                    "normalized_spacing": "NORMALIZE",
                    "already_spdx": "OK",
                }.get(action, action)
                print(f"  {label:>10}  {rel}")

    print()
    if args.dry_run:
        print("=== DRY RUN (no files modified) ===")
    print(f"  Replaced long-form:   {stats['replaced_long_form']}")
    print(f"  Added missing header: {stats['added_header']}")
    print(f"  Normalized spacing:   {stats['normalized_spacing']}")
    print(f"  Already correct:      {stats['already_spdx']}")
    print(f"  Skipped/errors:       {stats['skipped']}")
    print(f"  Total processed:      {sum(stats.values())}")


if __name__ == "__main__":
    main()
