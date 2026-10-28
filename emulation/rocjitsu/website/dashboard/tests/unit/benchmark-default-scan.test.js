import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, expect, test, vi } from 'vitest';
import * as selectors from '../../src/data/selectors.js';
import BenchmarksView from '../../src/components/views/BenchmarksView.jsx';
import { benchmarkData } from '../fixtures/publishedData.js';

const filters = { targets: benchmarkData.targets, suites: benchmarkData.suites, modes: benchmarkData.modes };
afterEach(() => vi.restoreAllMocks());

test.each([
  { selectedBenchmarks: [] },
  { selectedBenchmarks: benchmarkData.testCatalog.slice(0, 1) },
  { initialBenchmarkIds: [] },
])('explicit benchmark choices do not calculate unused default selections: %j', (selection) => {
  const defaults = vi.spyOn(selectors, 'selectDefaultBenchmarks');
  renderToStaticMarkup(createElement(BenchmarksView, { data: benchmarkData, filters, ...selection }));
  expect(defaults).not.toHaveBeenCalled();
});

test.each([{}, { selectedBenchmarks: null }])('unchosen benchmark selections still calculate measured defaults: %j', (selection) => {
  const defaults = vi.spyOn(selectors, 'selectDefaultBenchmarks');
  const html = renderToStaticMarkup(createElement(BenchmarksView, { data: benchmarkData, filters, ...selection }));
  expect(defaults).toHaveBeenCalledTimes(1);
  expect(html.match(/data-testid="benchmark-grid-card"/g)).toHaveLength(3);
});
