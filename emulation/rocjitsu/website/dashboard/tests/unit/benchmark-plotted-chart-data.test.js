import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, test, vi } from 'vitest';
import { themed } from '../helpers/themed.js';
import BenchmarkHistoryChart from '../../src/components/benchmarks/BenchmarkHistoryChart.jsx';
import BenchmarksView from '../../src/components/views/BenchmarksView.jsx';
import { benchmarkPlottedData, benchmarkRangeData } from '../../src/components/benchmarks/benchmarkExplorer.js';
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

// The published fixture covers only 24 commit days, so it cannot tell a 30-day window from a 90-day one. This history
// puts one run on each of days 0, 29, 30, 89 and 90 before the latest commit. The latest is late in its day and the
// day-29 and day-89 runs are early in theirs, so a window cut at the wrong hour or by one day too few or too many
// changes which runs a card receives.
const DAY = 86400000;
const LATEST = Date.parse('2026-10-04T22:00:00.000Z');
const atDay = (days, hour) => new Date(Date.parse('2026-10-04T00:00:00.000Z') - days * DAY + hour * 3600000).toISOString();
function historyAround() {
  const template = benchmarkData.runs.at(-1);
  const make = (runId, days, hour) => ({ ...structuredClone(template), runId, commitTimestamp: atDay(days, hour), timestamp: atDay(days, hour) });
  const runs = [make('day-90', 90, 1), make('day-89', 89, 1), make('day-30', 30, 12), make('day-29', 29, 1), make('day-0', 0, 22)];
  return { ...benchmarkData, runs, generatedAt: new Date(LATEST).toISOString() };
}
const idsFor = (range) => benchmarkPlottedData(historyAround(), range).runs.map((run) => run.runId).sort();

test('the ranges keep exactly the runs inside their trailing days', () => {
  expect(idsFor('1W')).toEqual(['day-0']);
  expect(idsFor('1M')).toEqual(['day-0', 'day-29']);
  expect(idsFor('3M')).toEqual(['day-0', 'day-29', 'day-30', 'day-89']);
  expect(idsFor('ALL')).toEqual(['day-0', 'day-29', 'day-30', 'day-89', 'day-90']);
});

// benchmarkRangeData is the first cut, before the day slots trim again, so a cutoff that is one day too wide shows here
// even when the plotted runs look right. day-30 starts 30 days before the latest day, one day outside the 1M window.
test('the range cutoff keeps the last 30 and 90 commit days and not the day after', () => {
  const keptBy = (range) => benchmarkRangeData(historyAround(), range).runs.map((run) => run.runId).sort();
  expect(keptBy('1M')).toEqual(['day-0', 'day-29']);
  expect(keptBy('3M')).toEqual(['day-0', 'day-29', 'day-30', 'day-89']);
});

test('1W plots the runs of the last seven commit days on the published fixture', () => {
  const plotted = benchmarkPlottedData(benchmarkData, '1W').runs.map((run) => run.runId);
  // One run per commit day in this fixture, so seven days are seven runs; on 1W a day with a rerun would add its runs too.
  expect(plotted).toEqual(['fictional-develop-17', 'fictional-develop-18', 'fictional-develop-19', 'fictional-develop-20', 'fictional-develop-21', 'fictional-develop-22', 'fictional-develop-23']);
  expect(plotted.length).toBeLessThan(benchmarkData.runs.length);
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
