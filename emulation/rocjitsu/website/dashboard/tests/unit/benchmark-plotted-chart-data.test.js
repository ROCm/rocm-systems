import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, test, vi } from 'vitest';
import { themed } from '../helpers/themed.js';
import BenchmarkHistoryChart from '../../src/components/benchmarks/BenchmarkHistoryChart.jsx';
import BenchmarksView from '../../src/components/views/BenchmarksView.jsx';
import { benchmarkPlottedData } from '../../src/components/benchmarks/benchmarkExplorer.js';
import { historyRanges } from '../../src/config/historyRanges.js';
import { benchmarkData } from '../fixtures/publishedData.js';

const captured = vi.hoisted(() => ({ option: null, cardData: [] }));
vi.mock('../../src/components/shared/Chart.jsx', () => ({ default: (props) => { captured.option = props.option; return createElement('div'); } }));
// Stands in for the chart only in the view test below, to see what data each card is handed.
vi.mock('../../src/components/benchmarks/BenchmarkHistoryChart.jsx', async (importOriginal) => {
  const original = await importOriginal();
  return { default: (props) => { captured.cardData.push(props.data); return original.default(props); } };
});

const filters = { targets: benchmarkData.targets, suites: benchmarkData.suites, modes: ['single', 'default'] };
const ids = benchmarkData.testCatalog.slice(0, 3).map((entry) => entry.id);
// Functions (tooltip formatters, symbol sizes) cannot be compared; the plotted data and styling can.
const plain = (option) => JSON.stringify(option.series, (key, value) => (typeof value === 'function' ? undefined : value));
const seriesFor = (data, id, range) => {
  renderToStaticMarkup(themed(BenchmarkHistoryChart, { data, filters, benchmark: { id, name: id }, range }));
  return plain(captured.option);
};

describe.each(['1W', '1M', '3M', 'ALL'])('%s range', (range) => {
  test.each(ids)('a card drawn from the plotted runs matches one drawn from the whole history (%s)', (id) => {
    expect(seriesFor(benchmarkPlottedData(benchmarkData, range), id, range)).toBe(seriesFor(benchmarkData, id, range));
  });
});

// The fixture covers 24 commit days (2026-09-11 to 2026-10-04) in 25 runs, so it can show the 7-day window cut and that
// 1M, 3M and ALL keep every day; it cannot tell 30 from 90 days. Those windows are pinned in historyRanges instead.
test('1W plots the runs of the last seven commit days, and 1M, 3M and ALL plot every day of a 24-day history', () => {
  const ids = (range) => benchmarkPlottedData(benchmarkData, range).runs.map((run) => run.runId);
  // One run per commit day in this fixture, so seven days are seven runs; on 1W a day with a rerun would add its runs too.
  expect(ids('1W')).toHaveLength(7);
  expect(ids('1W')).toEqual(['fictional-develop-17', 'fictional-develop-18', 'fictional-develop-19', 'fictional-develop-20', 'fictional-develop-21', 'fictional-develop-22', 'fictional-develop-23']);
  expect(benchmarkData.runs.length).toBeGreaterThan(ids('1W').length);
  for (const range of ['1M', '3M', 'ALL']) expect(ids(range)).toHaveLength(24);
});

test('the range windows stay 7, 30 and 90 days, with ALL unbounded', () => {
  expect(historyRanges.map(({ id, days }) => [id, days])).toEqual([['1W', 7], ['1M', 30], ['3M', 90], ['ALL', null]]);
});

test.each(['1W', '1M', '3M', 'ALL'])('%s: the view hands each card only the plotted runs, not the whole history', (range) => {
  captured.cardData = [];
  renderToStaticMarkup(themed(BenchmarksView, { data: benchmarkData, filters, historyRange: range, onRangeChange() {}, initialBenchmarkIds: ids }));
  expect(captured.cardData.length).toBeGreaterThan(0);
  const plotted = benchmarkPlottedData(benchmarkData, range).runs.map((run) => run.runId);
  for (const data of captured.cardData) expect(data.runs.map((run) => run.runId)).toEqual(plotted);
  // The 1W card is handed the seven runs of the last seven commit days (one per day here), not the 25 of the whole history.
  if (range === '1W') for (const data of captured.cardData) expect(data.runs).toHaveLength(7);
});
