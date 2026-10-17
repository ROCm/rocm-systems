import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import BenchmarksView from '../../src/components/views/BenchmarksView.jsx';
import * as explorer from '../../src/components/benchmarks/benchmarkExplorer.js';

const charts = vi.hoisted(() => ({ props: [] }));
vi.mock('../../src/components/benchmarks/BenchmarkHistoryChart.jsx', () => ({ default: (props) => { charts.props.push(props); return createElement('div', { 'data-testid': 'measured-chart' }); } }));
const catalog = ['a', 'b', 'c'].map((id) => ({ id, name: `Fictional workload ${id}`, suite: 'Triton', problem: {} }));
const data = { testCatalog: catalog, runs: [{ timestamp: '2026-10-05T14:00:00Z', tests: [
  { logicalTestId: 'a', target: 'gfx1250', mode: 'ST', suite: 'Triton', status: 'completed' },
  { logicalTestId: 'b', target: 'gfx1250', mode: 'MT', suite: 'Triton', status: 'failed' },
] }] };
const filters = { targets: ['gfx1250'], suites: ['Triton'], modes: ['ST'] };

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
  expect(html).toContain('ST · solid');
  expect(html).toContain('MT · dashed');
});

test('eligible catalog follows explicit mode and empty mode is an empty analysis', () => {
  expect(explorer.explorerCatalog(data, filters).available.map((test) => test.id)).toEqual(['a']);
  expect(explorer.explorerCatalog(data, { ...filters, modes: ['MT'] }).available.map((test) => test.id)).toEqual(['b']);
  expect(explorer.explorerCatalog(data, { ...filters, modes: [] }).available).toEqual([]);
});

test('workload range slices canonical source by commit period without changing measurements', () => {
  const runs = [1, 5, 30].map((day) => ({ runId: `fictional-${day}`, timestamp: `2026-09-${String(day).padStart(2, '0')}T12:00:00Z`, commitTimestamp: `2026-09-${String(day).padStart(2, '0')}T11:00:00Z`, tests: [] }));
  const ranged = explorer.benchmarkRangeData?.({ ...data, runs }, '1W');
  expect(ranged?.runs).toEqual([runs[2]]);
  expect(explorer.benchmarkRangeData?.({ ...data, runs }, 'ALL').runs).toEqual(runs);
});
