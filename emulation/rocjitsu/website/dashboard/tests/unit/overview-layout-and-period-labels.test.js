import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import MetricsGrid from '../../src/components/overview/MetricsGrid.jsx';
import OverviewView from '../../src/components/overview/OverviewView.jsx';
import { selectOverview } from '../../src/data/selectors.js';
import { benchmarkData } from '../fixtures/publishedData.js';

const filters = { targets: ['gfx1250'], suites: benchmarkData.suites, modes: benchmarkData.modes };
function renderOverview(scope = filters, range = 'ALL') {
  return renderToStaticMarkup(createElement(OverviewView, {
    data: benchmarkData,
    viewModel: selectOverview(benchmarkData, scope, range),
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
  expect(html).toContain('normalized to the latest test catalog');
});

test('normalization disclosure promises per-benchmark measurements, not removed run totals', () => {
  const html = renderOverview();
  expect(html).toContain('Explore original per-benchmark measurements');
  expect(html).not.toContain('Original run durations:');
});

test('missing results are not labeled measured failures', () => {
  const html = renderToStaticMarkup(createElement(MetricsGrid, { metrics: { total: 3, completed: 2, failed: 0, duration: null, durationDelta: null } }));
  expect(html).toContain('>Incomplete<');
  expect(html).not.toContain('>OK<');
});

test('metric captions name selected-period commit endpoints rather than all-history rerun chronology', () => {
  const html = renderOverview();
  expect(html).toContain('Selected period · all available history · first vs latest commit');
  expect(html).toContain('Sum of selected benchmark runtimes');
});

test('metric strip keeps selected-period meaning, original icon identities and compact OK health', () => {
  const metrics = selectOverview(benchmarkData, filters, 'ALL').metrics;
  const html = renderToStaticMarkup(createElement(MetricsGrid, { metrics }));
  expect(html).toContain('data-testid="metric-strip"');
  expect(html.match(/data-testid="metric-card-/g)).toHaveLength(4);
  expect(html).toContain('Selected period · all available history');
  expect(html).toContain('>OK<');
  expect(html).toContain('HealthAndSafetyRoundedIcon');
});
