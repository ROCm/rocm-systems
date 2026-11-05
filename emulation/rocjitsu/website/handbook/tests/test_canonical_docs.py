# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Verify canonical documentation publication and source-link rendering.
# Usage: python -m unittest discover -s tests

import importlib.util
from http.client import IncompleteRead
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from markdown import Markdown
import yaml

HANDBOOK = Path(__file__).resolve().parents[1]
SOURCE_URL = 'https://github.com/ROCm/rocm-systems/tree/develop/emulation/rocjitsu'
spec = importlib.util.spec_from_file_location('handbook_hooks', HANDBOOK / 'hooks.py')
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)


class RepositoryStatsTests(unittest.TestCase):
    def tearDown(self):
        hooks.repository_stats.cache_clear()

    def test_valid_counts_are_fetched_once_including_zero(self):
        with patch.object(
            hooks,
            'urlopen',
            return_value=io.StringIO('{"stargazers_count":0,"forks_count":42}'),
        ) as fetch:
            expected = {'stargazers_count': 0, 'forks_count': 42}
            self.assertEqual(hooks.repository_stats(), expected)
            self.assertEqual(hooks.repository_stats(), expected)
            fetch.assert_called_once()

    def test_invalid_responses_leave_an_offline_build_usable(self):
        for response in (
            'not json',
            'null',
            '[]',
            '{}',
            '{"stargazers_count":true,"forks_count":2}',
            '{"stargazers_count":-1,"forks_count":2}',
        ):
            with self.subTest(response=response):
                hooks.repository_stats.cache_clear()
                with patch.object(hooks, 'urlopen', return_value=io.StringIO(response)):
                    self.assertEqual(hooks.repository_stats(), {})

    def test_network_failure_leaves_an_offline_build_usable(self):
        for failure in (OSError('offline'), IncompleteRead(b'{')):
            with self.subTest(failure=failure):
                hooks.repository_stats.cache_clear()
                with patch.object(hooks, 'urlopen', side_effect=failure):
                    self.assertEqual(hooks.repository_stats(), {})


class SourceLinksTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.docs = self.root / 'rocjitsu' / 'docs'
        self.docs.mkdir(parents=True)
        (self.docs / 'architecture.md').write_text('# Architecture\n')
        (self.docs.parent / 'tools').mkdir()
        (self.docs.parent / 'tools' / 'run.py').write_text('# Source\n')
        (self.root / 'outside.md').write_text('# Outside Rocjitsu\n')
        extension = hooks.SourceLinksExtension(self.docs, SOURCE_URL)
        extension.source_path = self.docs / 'index.md'
        self.markdown = Markdown(extensions=[extension, 'fenced_code'])

    def test_existing_source_file_and_directory_links(self):
        html = self.markdown.convert(
            '[Source](../tools/run.py?plain=1#L2) [Directory](../tools/)'
        )
        self.assertIn(
            'href="https://github.com/ROCm/rocm-systems/blob/develop/'
            'emulation/rocjitsu/tools/run.py?plain=1#L2"',
            html,
        )
        self.assertIn(f'href="{SOURCE_URL}/tools"', html)

    def test_document_missing_and_outside_links_remain_unchanged(self):
        links = [
            'architecture.md#layers',
            '../missing.py',
            '../../outside.md',
            '#local-heading',
            'https://example.com/source.py',
        ]
        for link in links:
            with self.subTest(link=link):
                html = self.markdown.reset().convert(f'[Link]({link})')
                self.assertIn(f'href="{link}"', html)

    def test_nested_page_resolves_source_relative_to_page(self):
        nested = self.docs / 'guides'
        nested.mkdir()
        extension = hooks.SourceLinksExtension(self.docs, SOURCE_URL)
        extension.source_path = nested / 'guide.md'
        html = Markdown(extensions=[extension]).convert('[Source](../../tools/run.py)')
        self.assertIn(
            'href="https://github.com/ROCm/rocm-systems/blob/develop/'
            'emulation/rocjitsu/tools/run.py"',
            html,
        )

    def test_code_examples_are_not_rewritten(self):
        example = '[Source](../tools/run.py)'
        html = self.markdown.convert(f'`{example}`\n\n```markdown\n{example}\n```')
        self.assertIn(f'<code>{example}</code>', html)
        self.assertIn(f'<code class="language-markdown">{example}\n</code>', html)
        self.assertNotIn('github.com', html)


class CanonicalBuildTests(unittest.TestCase):
    def test_build_reads_canonical_docs_without_modifying_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'rocjitsu'
            docs = root / 'docs'
            handbook = root / 'website' / 'handbook'
            handbook.mkdir(parents=True)
            (docs / 'sphinx').mkdir(parents=True)
            readme = root / 'README.md'
            (root / 'CMakeLists.txt').write_text('# Build configuration\n')
            building = docs / 'building.md'
            building.write_text('# Building\n\nCANONICAL_FIRST_REVISION\n')
            (docs / 'sphinx' / 'private.md').write_text(
                '# Excluded\n\nSPHINX_ONLY_MARKER\n'
            )
            for directory in ('assets', 'overrides'):
                shutil.copytree(HANDBOOK / directory, handbook / directory)
            shutil.copy2(HANDBOOK / 'hooks.py', handbook / 'hooks.py')
            config = yaml.safe_load((HANDBOOK / 'mkdocs.yml').read_text())
            config['nav'] = [
                {'Overview': 'index.md'},
                {'Building': 'building.md'},
                {'Blog': 'blog/index.md'},
            ]
            (docs / 'blog' / 'posts' / 'nested').mkdir(parents=True)
            (docs / 'blog' / 'index.md').write_text('# Blog\n')
            for name, relative in (
                ('first', '../../../'),
                ('nested/second', '../../../../'),
            ):
                (docs / 'blog' / 'posts' / f'{name}.md').write_text(
                    f'---\ndate: 2025-01-01\n---\n\n# {Path(name).name}\n\n'
                    f'[Source]({relative}CMakeLists.txt?plain=1#L1)\n\n'
                    f'[Home]({relative}README.md#supported-architectures)\n\n'
                    '<!-- more -->\n\nFull post body.\n'
                )
            (handbook / 'mkdocs.yml').write_text(yaml.safe_dump(config))

            for marker in ('CANONICAL_FIRST_REVISION', 'CANONICAL_SECOND_REVISION'):
                with self.subTest(marker=marker):
                    readme.write_text(
                        f'# Project\n\n## Supported architectures\n\n{marker}\n\n'
                        '[![CI](https://github.com/ROCm/rocm-systems/actions/'
                        'workflows/rocjitsu-ci.yml/badge.svg)]'
                        '(https://github.com/ROCm/rocm-systems/actions)\n\n'
                        '[Build](docs/building.md#setup)\n'
                    )
                    readme_before = readme.read_bytes()
                    building.write_text(
                        f'# Building\n\n{marker}\n\n'
                        '[Source](../CMakeLists.txt)\n\n## Setup\n\n'
                        '[Support](../README.md#supported-architectures)\n'
                    )
                    before = {
                        path.relative_to(docs): path.read_bytes()
                        for path in docs.rglob('*')
                        if path.is_file()
                    }
                    result = subprocess.run(
                        [sys.executable, '-m', 'mkdocs', 'build', '--strict'],
                        cwd=handbook,
                        env={**os.environ, 'ROCJITSU_DOCS_OFFLINE': '1'},
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    site = handbook / 'site'
                    overview = (site / 'index.html').read_text()
                    self.assertIn(marker, overview)
                    self.assertNotIn('/badge.svg', overview)
                    self.assertIn('Material for MkDocs', overview)
                    self.assertIn('href="building/#setup"', overview)
                    self.assertFalse((site / 'project-overview').exists())
                    self.assertEqual(readme_before, readme.read_bytes())
                    page = (site / 'building' / 'index.html').read_text()
                    self.assertIn('href="../#supported-architectures"', page)
                    self.assertIn(marker, page)
                    self.assertIn('<h1 id="building"><span class="twemoji"', page)
                    self.assertIn('<h2 id="setup">Setup', page)
                    self.assertIn(
                        'href="https://github.com/ROCm/rocm-systems/blob/develop/'
                        'emulation/rocjitsu/CMakeLists.txt"',
                        page,
                    )
                    if marker == 'CANONICAL_SECOND_REVISION':
                        self.assertNotIn('CANONICAL_FIRST_REVISION', page)
                    search = (site / 'search' / 'search_index.json').read_text()
                    self.assertIn(marker, search)
                    self.assertNotIn('SPHINX_ONLY_MARKER', search)
                    for output, home, count in (
                        ('blog/index.html', '../#supported-architectures', 2),
                        ('blog/first/index.html', '../../#supported-architectures', 1),
                        ('blog/second/index.html', '../../#supported-architectures', 1),
                    ):
                        rendered = (site / output).read_text()
                        self.assertEqual(
                            rendered.count(
                                'href="https://github.com/ROCm/rocm-systems/blob/develop/'
                                'emulation/rocjitsu/CMakeLists.txt?plain=1#L1"'
                            ),
                            count,
                            output,
                        )
                        self.assertEqual(
                            rendered.count(f'href="{home}">Home</a>'), count, output
                        )
                        self.assertNotIn('href="../../../', rendered, output)
                    self.assertFalse((site / 'sphinx').exists())
                    self.assertEqual(
                        (site / 'assets' / 'handbook.css').read_bytes(),
                        (handbook / 'assets' / 'handbook.css').read_bytes(),
                    )
                    self.assertEqual(
                        before,
                        {
                            path.relative_to(docs): path.read_bytes()
                            for path in docs.rglob('*')
                            if path.is_file()
                        },
                    )


if __name__ == '__main__':
    unittest.main()
