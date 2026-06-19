#!/usr/bin/env python3
"""
scripts/link-checker.py — Validate internal Markdown links across docs.
Part of the docs-audit tooling.

Scope (what IS checked):
  - Relative file-existence of every `[text](path)` link in .md files
    under the search root (excluding dotfiles and ai_analysis/).
  - Skips links inside destructive-ignored paths (ai_analysis/, hidden).

Out of scope (what is NOT checked):
  - External HTTP/HTTPS URLs — skipped entirely (see is_external_url).
  - Anchor fragments (`#section-id`) — stripped before file-existence
    check. A broken anchor inside a valid file will NOT be flagged.
  See docs/known-issues.md for the rationale.

`--strict` flag:
  - Changes OUTPUT FORMAT only: in strict mode no human-readable
    preamble is printed; only CSV rows are emitted. The set of checks
    performed is identical in both modes.
"""

import re
import os
from pathlib import Path
from urllib.parse import urlparse
import sys

# Regex to match Markdown links: [text](url)
MARKDOWN_LINK_PATTERN = r'\[([^\]]+)\]\(([^)]+)\)'


def is_external_url(url):
    """Check if a URL is external (http/https)."""
    return url.startswith('http://') or url.startswith('https://')


def resolve_internal_link(doc_path, link_url):
    """Resolve a relative link from a document."""
    # Remove anchors
    link_path = link_url.split('#')[0]

    if not link_path:
        # Anchor-only link (same file)
        return str(doc_path)

    # If absolute (starts with /), resolve from repo root
    if link_path.startswith('/'):
        return link_path

    # Relative to document
    doc_dir = Path(doc_path).parent
    resolved = (doc_dir / link_path).resolve()
    return str(resolved)


def find_broken_links(search_root="."):
    """Scan all .md files for broken links."""
    search_root = Path(search_root)
    broken_links = []

    for md_file in search_root.rglob("*.md"):
        # Skip hidden and cache dirs
        if any(part.startswith('.') for part in md_file.parts):
            continue
        # Skip legacy ai_analysis tree — being deleted by the agentic refactor
        if 'ai_analysis' in md_file.parts:
            continue
        # Skip the upstream opencode submodule (MIT). Its .md files +
        # bun node_modules tree are third-party and out of our scope.
        if 'opencode' in md_file.parts or 'node_modules' in md_file.parts:
            continue

        try:
            content = md_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Warning: Could not read {md_file}: {e}", file=sys.stderr)
            continue

        for line_num, line in enumerate(content.split('\n'), 1):
            for match in re.finditer(MARKDOWN_LINK_PATTERN, line):
                text, link = match.groups()

                # Skip external URLs (best-effort)
                if is_external_url(link):
                    continue

                # Validate internal link
                link_path = link.split('#')[0]  # Remove anchor

                if not link_path:
                    # Anchor-only is OK
                    continue

                # Resolve relative to document
                if link_path.startswith('/'):
                    # Absolute path
                    abs_path = Path(search_root) / link_path.lstrip('/')
                else:
                    # Relative path
                    abs_path = (md_file.parent / link_path).resolve()

                # Check if file exists
                if not abs_path.exists():
                    broken_links.append({
                        'file': str(md_file.relative_to(search_root)),
                        'line': line_num,
                        'link': link,
                        'text': text,
                    })

    return broken_links


def print_csv(broken_links):
    """Print results as CSV."""
    print("file,line,link,text")
    for item in broken_links:
        # Escape CSV
        file_val = item['file'].replace('"', '""')
        link_val = item['link'].replace('"', '""')
        text_val = item['text'].replace('"', '""')
        print(f'"{file_val}",{item["line"]},"{link_val}","{text_val}"')


def main():
    """Main entry point."""
    # Parse args: optional directory + --strict flag
    strict = False
    args = [a for a in sys.argv[1:] if a != '--strict']
    if '--strict' in sys.argv:
        strict = True
    search_root = args[0] if args else "."

    broken_links = find_broken_links(search_root)

    if broken_links:
        if strict:
            # Strict mode — emit only CSV rows, no preamble
            print_csv(broken_links)
        else:
            print(f"Found {len(broken_links)} broken internal links:\n")
            print_csv(broken_links)
        return 1
    else:
        if not strict:
            print("✓ All internal links validated")
        return 0


if __name__ == "__main__":
    sys.exit(main())
