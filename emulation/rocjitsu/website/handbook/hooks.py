# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Publish canonical documentation with source links and presentation-only icons.
# Usage: mkdocs build --strict -f emulation/rocjitsu/website/handbook/mkdocs.yml

from functools import lru_cache
from http.client import HTTPException
import json
import os
from pathlib import Path
import posixpath
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import material
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from material.extensions import emoji
from mkdocs.structure.files import File
from pymdownx.superfences import fence_code_format

SECTION_ICONS = {
    "User guide": "material/book-open-variant",
    "Developer guide": "material/code-braces",
    "Project": "material/folder-outline",
}

PAGE_ICONS = {
    'building.md': 'hammer-wrench',
    'rocjitsu-cli.md': 'console',
    'configuration.md': 'cog-outline',
    'architecture.md': 'sitemap',
    'style.md': 'source-branch',
    'race-detector.md': 'bug-outline',
    'rocgdb-debugging.md': 'bug-outline',
    'benchmarking.md': 'chart-line',
    'simdojo.md': 'cog-outline',
    'vm-design.md': 'memory',
    'dbt-design.md': 'swap-horizontal',
    'dbi-design.md': 'magnify',
    'plugins.md': 'puzzle-outline',
    'performance-dashboard.md': 'chart-line',
    'blog/index.md': 'newspaper-variant-outline',
}


@lru_cache(maxsize=1)
def repository_stats():
    # Fetch once per build/preview process; an offline build remains usable.
    try:
        request = Request(
            'https://api.github.com/repos/ROCm/rocm-systems',
            headers={'User-Agent': 'rocjitsu-handbook'},
        )
        with urlopen(request, timeout=3) as response:
            data = json.load(response)
        keys = ('stargazers_count', 'forks_count')
        if isinstance(data, dict) and all(
            type(data.get(key)) is int and data[key] >= 0 for key in keys
        ):
            return {key: data[key] for key in keys}
    except (OSError, ValueError, HTTPException):
        pass
    return {}


class SourceLinksProcessor(Treeprocessor):
    def __init__(self, md, extension):
        super().__init__(md)
        self.extension = extension
        # Material retains excerpt renderers until after other pages render.
        self.source_path = extension.source_path
        self.page_uri = extension.page_uri

    def run(self, root):
        extension = self.extension
        # The README keeps its CI badge on GitHub, but the handbook omits it.
        if self.source_path == extension.docs_dir.parent / 'README.md':
            for paragraph in list(root):
                if paragraph.tag != 'p' or len(paragraph) != 1 or paragraph.text:
                    continue
                link = paragraph[0]
                if link.tag != 'a' or len(link) != 1 or link.text or link.tail:
                    continue
                image = link[0]
                source = urlsplit(image.get('src', ''))
                if (
                    image.tag == 'img'
                    and not image.tail
                    and source.netloc == 'github.com'
                    and '/actions/workflows/' in source.path
                    and source.path.endswith('/badge.svg')
                ):
                    root.remove(paragraph)
        for link in root.iter('a'):
            url = urlsplit(link.get('href', ''))
            if url.scheme or url.netloc or not url.path or url.path.startswith('/'):
                continue
            target = (self.source_path.parent / unquote(url.path)).resolve()
            if target in extension.published_sources:
                destination = posixpath.relpath(
                    extension.published_sources[target],
                    posixpath.dirname(self.page_uri) or '.',
                )
                link.set('href', urlunsplit(url._replace(path=destination)))
                continue
            if target.is_relative_to(extension.docs_dir):
                continue
            if (
                not target.is_relative_to(extension.docs_dir.parent)
                or not target.exists()
            ):
                continue
            relative = target.relative_to(extension.docs_dir.parent).as_posix()
            source_url = extension.source_url
            if target.is_file():
                source_url = source_url.replace('/tree/', '/blob/', 1)
            destination = urlsplit(f'{source_url}/{quote(relative)}')
            link.set(
                'href',
                urlunsplit(
                    destination._replace(query=url.query, fragment=url.fragment)
                ),
            )


class SourceLinksExtension(Extension):
    def __init__(self, docs_dir, source_url):
        super().__init__()
        self.docs_dir = Path(docs_dir).resolve()
        self.source_url = source_url.rstrip('/')
        self.source_path = self.docs_dir / 'index.md'
        self.page_uri = 'index.md'
        self.published_sources = {}

    def extendMarkdown(self, md):
        # Run before MkDocs (priority 0) and Material excerpts (priority 1)
        # resolve local links into output URLs.
        md.treeprocessors.register(SourceLinksProcessor(md, self), 'source_links', 2)


def on_config(config):
    config.extra['repository_stats'] = (
        {} if os.environ.get('ROCJITSU_DOCS_OFFLINE') == '1' else repository_stats()
    )
    # Keep Python callables here so mkdocs.yml remains standard, lintable YAML.
    config.mdx_configs['pymdownx.emoji'] = {
        'emoji_index': emoji.twemoji,
        'emoji_generator': emoji.to_svg,
    }
    config.mdx_configs['pymdownx.superfences'] = {
        'custom_fences': [
            {'name': 'mermaid', 'class': 'mermaid', 'format': fence_code_format},
        ],
    }
    config.markdown_extensions.append(
        SourceLinksExtension(config.docs_dir, config.repo_url)
    )
    return config


def on_files(files, config):
    readme = Path(config.docs_dir).parent / 'README.md'
    if readme.exists():
        files.append(File.generated(config, 'index.md', abs_src_path=str(readme)))
    for extension in config.markdown_extensions:
        if isinstance(extension, SourceLinksExtension):
            extension.published_sources = {
                Path(file.abs_src_path).resolve(): file.src_uri
                for file in files
                if file.is_documentation_page() and file.abs_src_path
            }
    for name in ('handbook.css', 'repository-stats.js'):
        asset = Path(__file__).parent / 'assets' / name
        files.append(File.generated(config, f'assets/{name}', abs_src_path=str(asset)))
    return files


def on_page_markdown(markdown, page, config, files):
    for extension in config.markdown_extensions:
        if isinstance(extension, SourceLinksExtension):
            extension.page_uri = page.file.src_uri
            extension.source_path = (
                Path(page.file.abs_src_path)
                if page.file.abs_src_path
                else Path(config.docs_dir) / page.file.src_uri
            )
    return markdown


def on_page_content(html, page, config, files):
    icon = PAGE_ICONS.get(page.file.src_uri)
    if icon:
        icon_path = (
            Path(material.__file__).parent
            / 'templates'
            / '.icons'
            / 'material'
            / f'{icon}.svg'
        )
        decoration = (
            f'<span class="twemoji" aria-hidden="true">{icon_path.read_text()}</span> '
        )
        html = re.sub(
            r'(<h1\b[^>]*>)', lambda match: match[1] + decoration, html, count=1
        )
    return html


def on_serve(server, config, builder):
    server.watch(str(Path(__file__).parent / 'assets'))
    server.watch(str(Path(config.docs_dir).parent / 'README.md'))
    return server


def on_nav(nav, **kwargs):
    for section in nav.items:
        if section.is_section and section.title in SECTION_ICONS:
            section.meta = {"icon": SECTION_ICONS[section.title]}
    return nav
