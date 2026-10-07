import { readFileSync } from 'node:fs';
import { createElement, isValidElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import BranchList from '../../src/components/branch/BranchList.jsx';
import BranchPicker from '../../src/components/branch/BranchPicker.jsx';
import BenchmarkDifferences from '../../src/components/branch/BenchmarkDifferences.jsx';
import BranchRunsView from '../../src/components/views/BranchRunsView.jsx';
import * as presentation from '../../src/components/branch/branchPresentation.js';
import * as fixtures from '../fixtures/publishedData.js';
import { selectPublishedBranches } from '../../src/data/branchSelectors.js';

const nodes = (element) => Array.isArray(element) ? element.flatMap(nodes) : !isValidElement(element) ? [] : [element, ...nodes(element.props.children)];
const render = (element) => renderToStaticMarkup(element);

test('Branches heading has no numeric status badge', () => {
  const list = BranchList({ branches: [], total: 20 });
  expect(nodes(list).filter((node) => node.props.role === 'status')).toEqual([]);
});

test('desktop picker has a named leftward collapse control without changing the route snapshot', () => {
  const html = render(createElement(BranchRunsView, { data: fixtures.benchmarkData, state: { branchSelection: {}, setBranchSelection() { throw new Error('UI collapse must not write selection'); } } }));
  expect(html).toContain('aria-label="Collapse branches"');
  expect(html).toContain('aria-expanded="true"');
  expect(html).toContain('aria-controls="branch-picker"');
  expect(html).toContain('ChevronLeft');
  expect(html).toContain('data-picker-collapsed="false"');
  const child = createElement('div', { 'data-testid': 'kept-mounted' }, 'selection/search state');
  const calls = [];
  for (const collapsed of [false, true, false]) {
    const tree = BranchPicker({ collapsed, onToggle: () => calls.push(!collapsed), children: child });
    const toggle = nodes(tree).find((node) => node.props['aria-controls'] === 'branch-picker');
    expect(toggle.props['aria-expanded']).toBe(!collapsed);
    expect(toggle.props['aria-label']).toBe(collapsed ? 'Expand branches' : 'Collapse branches');
    expect(nodes(tree)).toContain(child);
    toggle.props.onClick();
  }
  expect(calls).toEqual([true, false, true]);
});

test('branch search stays centered with a normal floating label and visible search prompt', () => {
  const html = render(createElement(BranchRunsView, { data: fixtures.benchmarkData, state: { branchSelection: {} } }));
  const toolbarClass = html.match(/class="branch-toolbar MuiBox-root (css-[^"]+)"/)[1];
  const rules = html.match(new RegExp(`\\.${toolbarClass}\\{([^}]+)\\}`))[1];
  expect(rules.includes('align-items:center')).toBe(true);
  const label = html.match(/<label[^>]*>Search branch, PR or SHA<\/label>/)[0];
  expect(label.includes('data-shrink="false"')).toBe(true);
  expect(html).toContain('placeholder="Search branch, PR or SHA"');
  expect(html).toContain('translate(14px, 12px) scale(1)');
  expect(html).toContain('padding-top:10px');
  expect(html).toContain('padding-bottom:10px');
});

test('exact requested fictional branch is schema-valid and preserves all original twenty branches', () => {
  expect(fixtures.publishedResult.data.schemaVersion).toBe(2);
  const branch = 'users/RattataKing/test-branch';
  const wire = fixtures.publishedRuns.find((run) => run.source.branch === branch);
  expect(wire).toBeDefined();
  expect(typeof wire.source.commit).toBe('string');
  expect(wire.source.commit).toMatch(/^[0-9a-f]{40}$/);
  expect(wire.source.base.branch).toBe('develop');
  expect(wire.source.base.commit).toMatch(/^[0-9a-f]{40}$/);
  expect(fixtures.publishedCatalogs[wire.testCatalog]).toBeDefined();
  expect(selectPublishedBranches(fixtures.benchmarkData)).toHaveLength(21);
  for (let index = 1; index <= 20; index++) expect(fixtures.publishedRuns.some((run) => run.source.branch === `fictional/optimization-${String(index).padStart(2, '0')}`)).toBe(true);
  const generated = fixtures.createFeedbackPublication();
  expect(generated.runs.find((run) => run.source.branch === branch)).toEqual(wire);
  expect(generated.index).toEqual(fixtures.dataIndex);
  const checkedIn = JSON.parse(readFileSync(new URL('../fixtures/data/runs/fictional-rattataking-test-branch.json', import.meta.url), 'utf8'));
  expect(checkedIn).toEqual(wire);
});

const pair = (id, base, next) => ({ candidateTest: { testId: id, name: id, suite: 'Suite', durationSeconds: next }, baselineTest: { durationSeconds: base }, delta: base ? (next - base) / base * 100 : null });
const ids = (rows, key, direction) => presentation.groupComparisonRows(rows, key, direction).flatMap((group) => group.rows.map((row) => row.candidateTest.testId));

test('Abs change sorts signed duration separately from Pct change, with missing last and deterministic ties both ways', () => {
  const rows = [pair('negative', 100, 80), pair('positive', 1, 2), pair('zero', 0, 0), pair('a-tie', 5, 6), pair('missing', null, 2)];
  expect(ids(rows, 'seconds', 'desc')).toEqual(['a-tie', 'positive', 'zero', 'negative', 'missing']);
  expect(ids(rows, 'seconds', 'asc')).toEqual(['negative', 'zero', 'a-tie', 'positive', 'missing']);
  expect(ids(rows, 'percent', 'desc')).toEqual(['positive', 'a-tie', 'negative', 'missing', 'zero']);
  expect(ids(rows, 'percent', 'asc')).toEqual(['negative', 'a-tie', 'positive', 'missing', 'zero']);
  expect(presentation.rowChanges(pair('zero-reference', 0, 2))).toEqual({ seconds: 2, percent: null });
  expect(presentation.rowChanges(pair('zero', 0, 0))).toEqual({ seconds: 0, percent: null });
  expect(presentation.rowChanges(pair('missing', null, 0))).toEqual({ seconds: null, percent: null });
});

test('sorting remains monotonic across real suite boundaries, including unavailable values', () => {
  const rows = [pair('negative', 100, 80), { ...pair('positive', 1, 2), candidateTest: { ...pair('positive', 1, 2).candidateTest, suite: 'Other' } }, pair('zero', 0, 0), { ...pair('missing', null, 1), candidateTest: { ...pair('missing', null, 1).candidateTest, suite: 'Other' } }];
  expect(ids(rows, 'seconds', 'desc')).toEqual(['positive', 'zero', 'negative', 'missing']);
  expect(ids(rows, 'seconds', 'asc')).toEqual(['negative', 'zero', 'positive', 'missing']);
});

test('separate sortable change headers expose active aria-sort and emit both directions; mobile retains sort controls', () => {
  const calls = [];
  const props = { comparison: { comparable: [pair('negative', 100, 80), pair('positive', 1, 2)], notComparable: [], baselineDuration: 101, candidateDuration: 82 }, summary: { available: true, matched: 2, excluded: 0 },
    candidate: { configurations: [] }, reference: { configurations: [] }, selection: { target: 'gfx1250', mode: 'ST' }, suites: ['Suite'], suiteOptions: ['Suite'], query: '', onSort: (sort) => calls.push(sort),
  };
  for (const sort of [{ key: 'seconds', direction: 'desc' }, { key: 'seconds', direction: 'asc' }, { key: 'percent', direction: 'desc' }]) {
    const tree = BenchmarkDifferences({ ...props, sort });
    const headers = nodes(tree).filter((node) => /^branch-(mobile-)?sort-/.test(node.props['data-testid'] ?? ''));
    expect(headers).toHaveLength(4);
    for (const key of ['seconds', 'percent']) {
      const header = headers.find((node) => node.props['data-testid'] === `branch-sort-${key}`);
      expect(header.props.active).toBe(sort.key === key);
      header.props.onClick();
      expect(calls.at(-1)).toEqual({ key, direction: sort.key === key && sort.direction === 'desc' ? 'asc' : 'desc' });
    }
    const html = render(tree);
    expect(html.includes('Sort by magnitude')).toBe(false);
    expect(html.includes('Abs change')).toBe(true);
    expect(html.includes('Pct change')).toBe(true);
    expect(html.includes(`aria-sort="${sort.direction === 'asc' ? 'ascending' : 'descending'}"`)).toBe(true);
    expect(html.includes('branch-mobile-sort-seconds')).toBe(true);
  }
});
