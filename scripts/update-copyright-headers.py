#!/usr/bin/env python3
"""
Update old-style NVIDIA copyright headers to NCCL's SPDX format.

Old-style C/C++ header:
  /*************************************************************************
   * Copyright (c) YYYY[-ZZZZ], NVIDIA CORPORATION. All rights reserved.
   * [Modifications Copyright (c) [YEAR] Advanced Micro Devices, Inc. ...]
   *
   * See LICENSE.txt for license information
   ************************************************************************/

New SPDX header:
  /*************************************************************************
   * SPDX-FileCopyrightText: Copyright (c) YYYY-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   * SPDX-License-Identifier: Apache-2.0
   *
   * [Modifications Copyright (c) YEAR Advanced Micro Devices, Inc. All rights reserved.]
   *
   * See LICENSE.txt for more license information
   *************************************************************************/

Usage:
    python3 scripts/update-copyright-headers.py [file1 file2 ...]
    python3 scripts/update-copyright-headers.py --dry-run [file1 ...]

If no files are given, reads from stdin (one path per line).
"""

import re
import sys
import os
import argparse
from pathlib import Path

# Regex for old-style C/C++ header block
# Group 1: start year (e.g. "2015" or "2015-2022" -> we keep the START year)
# Group 2: optional AMD modification line (full line content)
OLD_HEADER_RE = re.compile(
    r'(/\*+\n'                                   # opening /*...
    r' \* Copyright \(c\) (\d{4})(?:-\d{4})?, NVIDIA CORPORATION\. All rights reserved\.\n'
    r'(?: \* Modification[s]? Copyright \(c\).*?Advanced Micro Devices.*?\n)?'  # optional AMD line
    r' \*\n'
    r' \* See LICENSE\.txt for license information\n'
    r' \*+/)',
    re.MULTILINE
)

CURRENT_YEAR = 2026

def extract_start_year(content: str) -> str:
    """Extract the start year from the NVIDIA copyright line."""
    m = re.search(r'Copyright \(c\) (\d{4})(?:-\d{4})?, NVIDIA CORPORATION', content)
    if m:
        return m.group(1)
    return str(CURRENT_YEAR)

def extract_amd_line(content: str) -> str | None:
    """Return the existing AMD modification line text, or None."""
    m = re.search(r'( \* Modification[s]? Copyright \(c\).*?Advanced Micro Devices.*?\n)', content)
    if m:
        return m.group(1).rstrip('\n')
    return None

def build_new_header(start_year: str, amd_line: str | None) -> str:
    """Build the new SPDX header block."""
    # Opening:  /*  + 73 '*' = 74 chars  (matches existing RCCL SPDX files)
    # Closing:  ' ' + 73 '*' + '/' = 75 chars  (trailing '*' per todo spec)
    open_line  = '/' + '*' * 73
    close_line = ' ' + '*' * 73 + '/'  # trailing * per todo

    nvidia_line = (
        f' * SPDX-FileCopyrightText: Copyright (c) {start_year}-{CURRENT_YEAR} '
        f'NVIDIA CORPORATION & AFFILIATES. All rights reserved.'
    )
    spdx_id    = ' * SPDX-License-Identifier: Apache-2.0'
    license_ln = ' * See LICENSE.txt for more license information'

    lines = [open_line, nvidia_line, spdx_id, ' *']
    if amd_line:
        lines.append(amd_line)
        lines.append(' *')
    lines.append(license_ln)
    lines.append(close_line)

    return '\n'.join(lines)

# Pattern 1: long /****/ header with "See LICENSE.txt for license information"
LONG_HEADER_RE = re.compile(
    r'/\*+\n'
    r' \* Copyright \(c\) (\d{4})(?:-\d{4})?, NVIDIA CORPORATION\. All rights reserved\.\n'
    r'(?: \* Modification[s]? Copyright \(c\).*?Advanced Micro Devices.*?\n)?'
    r' \*\n'
    r' \* See LICENSE\.txt for license information\n'
    r' \*+/',
    re.MULTILINE
)

# Pattern 2: short /* Copyright ... */ with NO "See LICENSE" line
SHORT_HEADER_RE = re.compile(
    r'/\*\n'
    r' \* Copyright \(c\) (\d{4})(?:-\d{4})?, NVIDIA CORPORATION\. All rights reserved\.\n'
    r'(?: \* Modification[s]? Copyright \(c\).*?Advanced Micro Devices.*?\n)?'
    r' \*/',
    re.MULTILINE
)


def convert_file(path: str, dry_run: bool = False) -> bool:
    """
    Convert one file in place.  Returns True if the file was (or would be) changed.
    """
    text = Path(path).read_text(encoding='utf-8', errors='replace')

    # Try long header first, then short
    m = LONG_HEADER_RE.search(text)
    if not m:
        m = SHORT_HEADER_RE.search(text)

    if not m:
        return False

    old_block = m.group(0)
    start_year = extract_start_year(old_block)
    amd_line   = extract_amd_line(old_block)

    # Detect actual opening line length (to preserve style)
    open_len = old_block.index('\n')  # length of first line e.g. /***...
    open_stars = open_len  # number of chars on first line

    new_header = build_new_header(start_year, amd_line)

    new_text = text[:m.start()] + new_header + text[m.end():]

    if new_text == text:
        return False

    if dry_run:
        print(f"[DRY-RUN] Would convert: {path}")
        print(f"  start_year={start_year}, amd_line={amd_line!r}")
        return True

    Path(path).write_text(new_text, encoding='utf-8')
    print(f"Converted: {path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', help='Files to convert')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be changed without writing')
    args = parser.parse_args()

    files = args.files
    if not files:
        files = [line.strip() for line in sys.stdin if line.strip()]

    changed = 0
    skipped = 0
    for f in files:
        if not os.path.isfile(f):
            print(f"WARNING: not a file: {f}", file=sys.stderr)
            continue
        if convert_file(f, dry_run=args.dry_run):
            changed += 1
        else:
            skipped += 1

    print(f"\nDone. Changed={changed}, Unchanged/skipped={skipped}")


if __name__ == '__main__':
    main()
