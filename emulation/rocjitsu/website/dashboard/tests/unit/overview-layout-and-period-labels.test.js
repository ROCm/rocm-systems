import { renderToStaticMarkup } from 'react-dom/server';
import { themed } from '../helpers/themed.js';
import { expect, test } from 'vitest';
import OverviewView from '../../src/components/views/OverviewView.jsx';
import { selectOverviewProjection } from '../../src/data/overviewProjection.js';
import { benchmarkData } from '../fixtures/publishedData.js';

const filters = { targets: ['gfx1250'], suites: benchmarkData.suites, modes: benchmarkData.modes };
function renderOverview(scope = filters, range = 'ALL', catalogId = null) {
  return renderToStaticMarkup(themed(OverviewView, {
    data: benchmarkData,
    viewModel: selectOverviewProjection(benchmarkData, scope, { range, catalogId }),
    state: { filters: scope, historyRange: range, setHistoryRange() {}, search: '', setSearch() {} },
    onCompareRun() {}, onExploreRun() {}, onOpenBenchmarks() {},
  }));
}

test('Overview omits the secondary latest-results surface', () => {
  const html = renderOverview();
  expect(html.toLowerCase()).toContain('Performance Trend'.toLowerCase());
  expect(html.toLowerCase()).toContain('Largest Changes'.toLowerCase());
  expect(html.toLowerCase()).toContain('Recent Runs'.toLowerCase());
  expect(html).not.toContain('Latest Commit Results');
});


test('trend exposes only reference ranges with normalization disclosure', () => {
  const html = renderOverview();
  for (const label of ['Trailing 7 days', 'Trailing 30 days', 'Trailing 90 days', 'All available history']) expect(html).toContain(`aria-label="${label}"`);
  for (const label of ['Trailing 24 hours', 'Trailing 180 days', 'Year to date']) expect(html).not.toContain(`aria-label="${label}"`);
  expect(html).toContain('1.00× = each configuration');
});
