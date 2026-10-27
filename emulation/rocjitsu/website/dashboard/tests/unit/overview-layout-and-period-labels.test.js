import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import OverviewView from '../../src/components/views/OverviewView.jsx';
import { selectOverviewProjection } from '../../src/data/overviewProjection.js';
import { benchmarkData } from '../fixtures/publishedData.js';

const filters = { targets: ['gfx1250'], suites: benchmarkData.suites, modes: benchmarkData.modes };
function renderOverview(scope = filters, range = 'ALL', catalogId = null) {
  return renderToStaticMarkup(createElement(OverviewView, {
    data: benchmarkData,
    viewModel: selectOverviewProjection(benchmarkData, scope, { range, catalogId }),
    state: { filters: scope, historyRange: range, setHistoryRange() {}, search: '', setSearch() {} },
    onCompareRun() {}, onExploreRun() {}, onOpenBenchmarks() {},
  }));
}

test('Overview omits the secondary latest-results surface', () => {
  const html = renderOverview();
  expect(html).toContain('Performance Trend');
  expect(html).toContain('Largest Changes');
  expect(html).toContain('Recent Runs');
  expect(html).not.toContain('Latest Commit Results');
});

test('recent runs omit auxiliary controls but keep timestamps and explicit empty scope', () => {
  const html = renderOverview();
  expect(html).not.toContain('Show runs');
  expect(html).not.toContain('View in Explorer');
  expect(html).not.toContain('aria-label="Compare ');
  expect(html).toContain('dateTime=');
  expect(renderOverview({ targets: [], suites: [], modes: [] })).toContain('No selected tests');
});

test('trend exposes only reference ranges with normalization disclosure', () => {
  const html = renderOverview();
  for (const label of ['Trailing 7 days', 'Trailing 30 days', 'Trailing 90 days', 'All available history']) expect(html).toContain(`aria-label="${label}"`);
  for (const label of ['Trailing 24 hours', 'Trailing 180 days', 'Year to date']) expect(html).not.toContain(`aria-label="${label}"`);
  expect(html).toContain('1.00× = each configuration');
});

test('overview exposes compact catalog scope without anchor estimates', () => {
  const html = renderOverview();
  expect(html).toContain('Catalog workloads details');
  expect(html).toContain('General benchmark explorer (uses its own filters)');
  expect(html).not.toContain('first-success anchors');
  expect(html.match(/data-testid="metric-card-/g)).toHaveLength(4);
});

test('catalog labels expose exact names, truthful date context and unavailable identity', () => {
  const known = renderOverview(filters, 'ALL', 'fictional-old');
  expect(known).toContain('fictional-old');
  expect(known).toContain('Latest published develop commit using this benchmark set (UTC)');
  expect(known).not.toMatch(/Catalog \d+ ·/);
  const unknown = renderOverview(filters, '1M', 'historical-hash');
  expect(unknown).toContain('Unavailable benchmark set: historical-hash');
  expect(unknown.match(/data-testid="metric-card-/g)).toHaveLength(4);
});
