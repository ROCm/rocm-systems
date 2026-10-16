# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Check that publishing existing guides preserves their structural meaning.
# Usage: python -m unittest discover -s tests

from html.parser import HTMLParser
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from xml.etree.ElementTree import Element, SubElement

HANDBOOK = Path(__file__).resolve().parents[1]


class RenderedLists(HTMLParser):
    """Keep list hierarchy and each item's complete visible text."""

    def __init__(self, html):
        super().__init__()
        self.root = Element('root')
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag in ('ol', 'ul', 'li'):
            self.stack.append(SubElement(self.stack[-1], tag))

    def handle_endtag(self, tag):
        if tag in ('ol', 'ul', 'li'):
            self.stack.pop()

    def handle_data(self, data):
        for node in self.stack[1:]:
            node.text = (node.text or '') + data


class GuideRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.site = Path(temporary.name) / 'site'
        result = subprocess.run(
            [
                sys.executable,
                '-m',
                'mkdocs',
                'build',
                '--strict',
                '--site-dir',
                str(cls.site),
            ],
            cwd=HANDBOOK,
            env={**os.environ, 'ROCJITSU_DOCS_OFFLINE': '1'},
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def page(self, name):
        return (self.site / name / 'index.html').read_text()

    def test_race_checks_belong_to_step_five_of_six(self):
        lists = RenderedLists(self.page('race-detector'))
        steps = [
            node
            for node in lists.root.iter('ol')
            if 'Wave 0 executes ds_write_b32' in (node.text or '')
        ]
        self.assertEqual(len(steps), 1)
        items = steps[0].findall('li')
        self.assertEqual(len(items), 6)
        self.assertIn('Wave 0 executes ds_read_b32', items[4].text)
        checks = items[4].findall('./ul/li')
        self.assertEqual(len(checks), 3)
        for item, label in zip(checks, ('Fast path', 'Slow path', 'Race reported.')):
            self.assertIn(label, item.text)
        self.assertIn('What s_barrier would fix', items[5].text)

    def test_lists_after_introductory_prose_are_list_items(self):
        for page, marker in (
            ('vm-design', 'kernel_entry_pc'),
            ('vm-design', 's_gl1_inv'),
            ('dbi-design', 'VGPRs — a direct'),
            ('dbi-design', 'anchor_offset is dword aligned.'),
            ('dbi-design', 'build_scratch_store_dword / build_scratch_load_dword'),
            ('rocjitsu-cli', 'RPC_IOCTL (ALLOC_MEMORY response)'),
            ('rocjitsu-cli', 'UnixTransport::listen(endpoint)'),
            ('rocjitsu-cli', 'Unix: Unix domain socket + SCM_RIGHTS'),
        ):
            with self.subTest(page=page, marker=marker):
                lists = RenderedLists(self.page(page))
                self.assertTrue(
                    any(marker in (li.text or '') for li in lists.root.iter('li')),
                    f'{marker} must be inside a list item',
                )

    def test_continuation_paragraphs_stay_in_their_list_item(self):
        for page, first, continuation, next_item in (
            (
                'dbi-overview',
                'Sampling',
                'Finer granularity',
                'Filtering in the probe.',
            ),
            (
                'dbi-overview',
                'How long it must live.',
                'Completion is observable',
                'Read from the dispatch packet.',
            ),
            (
                'rocjitsu_dbt_guest',
                'HSA_TOOLS_LIB',
                'The supported DBT launch path',
                'HSA_TOOLS_DISABLE_REGISTER=1',
            ),
        ):
            with self.subTest(page=page):
                lists = RenderedLists(self.page(page))
                items = [
                    li
                    for li in lists.root.iter('li')
                    if continuation in (li.text or '')
                ]
                self.assertEqual(len(items), 1)
                self.assertIn(first, items[0].text)
                self.assertNotIn(next_item, items[0].text)

    def test_numbered_principles_remain_single_lists(self):
        for page, marker, count, item_index, continuation in (
            ('dbi-overview', 'Transparency.', 7, 0, 'This extends to state'),
            (
                'rocjitsu_dbt_guest',
                'ROCR must internally discover',
                5,
                1,
                'Calls that applications use',
            ),
        ):
            with self.subTest(page=page):
                lists = RenderedLists(self.page(page))
                ordered = [
                    node
                    for node in lists.root.iter('ol')
                    if marker in (node.text or '')
                ]
                self.assertEqual(len(ordered), 1)
                items = ordered[0].findall('li')
                self.assertEqual(len(items), count)
                self.assertIn(continuation, items[item_index].text)

    def test_template_names_are_literal_code(self):
        html = self.page('simdojo')
        for name, parameters in (
            ('Clocked', 'Base'),
            ('Functional', 'Base'),
            ('ExecBase', 'Mode, Base'),
            ('Cache', 'NumSets, Assoc, LineSizeBits'),
            ('VectorReg', 'N, T'),
        ):
            with self.subTest(name=name):
                expected = f'<code>{name}&lt;{parameters}&gt;</code>'
                self.assertTrue(expected in html, expected)
                self.assertFalse(f'{name}\\' in html, f'Escaped template: {name}')


if __name__ == '__main__':
    unittest.main()
