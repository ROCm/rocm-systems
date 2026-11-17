import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import BenchmarksView from '../../src/components/views/BenchmarksView.jsx';
import * as explorer from '../../src/components/benchmarks/benchmarkExplorer.js';
import { selectBenchmarkCatalog } from '../../src/data/selectors.js';
import { selectDaySlots } from '../../src/data/overviewProjection.js';

const charts = vi.hoisted(() => ({ props: [], plottedRanges: [] }));
vi.mock('../../src/components/benchmarks/benchmarkExplorer.js', async (importOriginal) => {
  const original = await importOriginal();
  return { ...original, benchmarkPlottedData: (source, range) => { charts.plottedRanges.push(range); return original.benchmarkPlottedData(source, range); } };
});
vi.mock('../../src/components/benchmarks/BenchmarkHistoryChart.jsx', () => ({ default: (props) => { charts.props.push(props); return createElement('div', { 'data-testid': 'measured-chart' }); } }));
const catalog = ['a', 'b', 'c'].map((id) => ({ id, name: `Fictional workload ${id}`, suite: 'Triton', problem: {} }));
const data = { testCatalog: catalog, runs: [{ timestamp: '2026-10-05T14:00:00Z', tests: [
  { logicalTestId: 'a', target: 'gfx1250', mode: 'single', suite: 'Triton', status: 'completed' },
  { logicalTestId: 'b', target: 'gfx1250', mode: 'default', suite: 'Triton', status: 'failed' },
] }] };
const filters = { targets: ['gfx1250'], suites: ['Triton'], modes: ['single'] };

test('grid-first page has shared range, Add and inspect controls while retaining excluded workload identity', () => {
  charts.props = [];
  const html = renderToStaticMarkup(createElement(BenchmarksView, { data, filters, historyRange: '1M', onRangeChange() {}, initialBenchmarkIds: ['a', 'b'] }));
  expect(html).not.toContain('Benchmark display mode');
  expect(html).toContain('aria-label="Benchmark history timeframe"');
  expect(html).toContain('Add benchmarks');
  expect(html).toContain('Fictional workload b');
  expect(html).toContain('Not in this configuration');
  expect(html).toContain('Inspect results');
  expect(html.match(/data-testid="benchmark-grid-card"/g)).toHaveLength(2);
  expect(charts.props).toHaveLength(1);
  expect(charts.props[0].range).toBe('1M');
  expect(html).toContain('Remove Fictional workload b from grid');
  expect(html).toContain('aria-label="Chart category and thread policy legend"');
});

test('eligible catalog follows explicit mode and empty mode is an empty analysis', () => {
  expect(selectBenchmarkCatalog(data, filters).available.map((test) => test.id)).toEqual(['a']);
  expect(selectBenchmarkCatalog(data, { ...filters, modes: ['default'] }).available.map((test) => test.id)).toEqual(['b']);
  expect(selectBenchmarkCatalog(data, { ...filters, modes: [] }).available).toEqual([]);
});

test('workload range slices canonical source by commit period without changing measurements', () => {
  const runs = [1, 5, 30].map((day) => ({ runId: `fictional-${day}`, timestamp: `2026-09-${String(day).padStart(2, '0')}T12:00:00Z`, commitTimestamp: `2026-09-${String(day).padStart(2, '0')}T11:00:00Z`, tests: [] }));
  const ranged = explorer.benchmarkRangeData?.({ ...data, runs }, '1W');
  expect(ranged?.runs).toEqual([runs[2]]);
  expect(explorer.benchmarkRangeData?.({ ...data, runs }, 'ALL').runs).toEqual(runs);
});

test.each(['1M', '3M', 'ALL', '1W'])('%s: Inspect results offers exactly the runs the chart plots on a day with a completed then a failed run', (range) => {
  const day = (runId, timestamp, status) => ({ runId, timestamp, commitTimestamp: '2026-10-05T09:00:00Z', provenance: { rocjitsuCommitSha: 'abc1234' },
    tests: [{ logicalTestId: 'a', target: 'gfx1250', mode: 'single', suite: 'Triton', status, durationSeconds: status === 'completed' ? 2 : null }] });
  const runs = [day('earlier-completed', '2026-10-05T10:00:00Z', 'completed'), day('later-failed', '2026-10-05T12:00:00Z', 'failed')];
  const source = { ...data, runs, generatedAt: '2026-10-05T14:00:00Z' };
  const plotted = selectDaySlots(source.runs, source.generatedAt, range).slots.flatMap((slot) => slot.run ? [slot.run.runId] : []);
  expect(explorer.benchmarkPlottedData(source, range).runs.map((run) => run.runId)).toEqual(plotted);
  expect(plotted).toEqual(range === '1W' ? ['earlier-completed', 'later-failed'] : ['later-failed']);
});

test.each(['1W', '1M', '3M', 'ALL'])('%s: the view builds the chart and Inspect results from the plotted-run helper', (range) => {
  charts.plottedRanges = [];
  renderToStaticMarkup(createElement(BenchmarksView, { data, filters, historyRange: range, onRangeChange() {}, initialBenchmarkIds: ['a'] }));
  expect(charts.plottedRanges).toContain(range);
});
