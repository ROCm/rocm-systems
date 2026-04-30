#!/usr/bin/env python3
"""Validate internal Markdown links across the PerfXpert docs tree."""

import re
from pathlib import Path
from urllib.parse import unquote
import sys

# Regex to match Markdown links: [text](url)
MARKDOWN_LINK_PATTERN = r'\[([^\]]+)\]\(([^)]+)\)'
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
HTML_ID_PATTERN = re.compile(r"""<[^>]+\s(?:id|name)=["']([^"']+)["']""")
EXPLICIT_ANCHOR_PATTERN = re.compile(r"\{#([A-Za-z0-9_.:-]+)\}\s*$")


def is_external_url(url):
    """Check if a URL is external (http/https)."""
    return url.startswith('http://') or url.startswith('https://')


def _github_slug(text):
    """Return the GitHub-style slug for a Markdown heading."""
    text = EXPLICIT_ANCHOR_PATTERN.sub("", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("`", "")
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _markdown_anchors(path):
    """Collect heading and explicit HTML anchors from a Markdown file."""
    anchors = set()
    counts = {}
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return anchors

    for line in content.splitlines():
        for match in HTML_ID_PATTERN.finditer(line):
            anchors.add(match.group(1))

        heading = HEADING_PATTERN.match(line)
        if not heading:
            continue
        title = heading.group(2)
        explicit = EXPLICIT_ANCHOR_PATTERN.search(title)
        if explicit:
            anchors.add(explicit.group(1))
        slug = _github_slug(title)
        if not slug:
            continue
        seen = counts.get(slug, 0)
        anchors.add(slug if seen == 0 else f"{slug}-{seen}")
        counts[slug] = seen + 1

    return anchors


def _split_link(link):
    """Split a Markdown link target into path and decoded fragment."""
    if "#" not in link:
        return link, ""
    path, fragment = link.split("#", 1)
    return path, unquote(fragment)


def _resolve_link_path(search_root, md_file, link_path):
    if not link_path:
        return md_file
    if link_path.startswith('/'):
        return Path(search_root) / link_path.lstrip('/')
    return (md_file.parent / link_path).resolve()


def _anchor_target_path(path):
    if path.is_dir():
        return path / "README.md"
    return path


def find_broken_links(search_root=".", *, validate_anchors=False):
    """Scan all .md files for broken links."""
    search_root = Path(search_root)
    broken_links = []
    anchor_cache = {}

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

                link_path, fragment = _split_link(link)
                abs_path = _resolve_link_path(search_root, md_file, link_path)

                # Check if file exists
                if not abs_path.exists():
                    broken_links.append({
                        'file': str(md_file.relative_to(search_root)),
                        'line': line_num,
                        'link': link,
                        'text': text,
                    })
                    continue

                if not validate_anchors or not fragment:
                    continue

                anchor_target = _anchor_target_path(abs_path)
                if anchor_target.suffix.lower() != ".md" or not anchor_target.exists():
                    continue

                anchors = anchor_cache.setdefault(anchor_target, _markdown_anchors(anchor_target))
                if fragment not in anchors:
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

    broken_links = find_broken_links(search_root, validate_anchors=strict)

    if broken_links:
        if strict:
            # Strict mode — emit only CSV rows, no preamble
            print_csv(broken_links)
        else:
            print(f"Found {len(broken_links)} broken internal links or anchors:\n")
            print_csv(broken_links)
        return 1
    else:
        if not strict:
            print("✓ All internal links validated")
        return 0


if __name__ == "__main__":
    sys.exit(main())
