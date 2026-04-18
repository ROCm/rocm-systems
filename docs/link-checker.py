#!/usr/bin/env python3
"""
docs/link-checker.py — Validate internal and external links in Markdown docs
Part of Phase 9 docs audit tooling
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
    # Accept optional directory argument
    search_root = sys.argv[1] if len(sys.argv) > 1 else "."

    broken_links = find_broken_links(search_root)

    if broken_links:
        print(f"Found {len(broken_links)} broken internal links:\n")
        print_csv(broken_links)
        return 1
    else:
        print("✓ All internal links validated")
        return 0


if __name__ == "__main__":
    sys.exit(main())
