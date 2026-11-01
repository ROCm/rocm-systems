import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { themed } from '../helpers/themed.js';
import { expect, test, vi } from 'vitest';
import BenchmarkHistoryChart from '../../src/components/benchmarks/BenchmarkHistoryChart.jsx';
const captured = vi.hoisted(() => ({ chart: null, data: null, model: null }));
vi.mock('../../src/components/shared/Chart.jsx', () => ({ default: (props) => { captured.chart = props; return createElement('div'); } }));
vi.mock('../../src/data/selectors.js', () => ({ selectBenchmarkSeries: (data) => { captured.data = data; return captured.model; } }));

function setup() {
  const runs = [1, 30].map((day) => ({ runId: `fictional-${day}`, timestamp: `2026-09-${String(day).padStart(2, '0')}T12:00:00Z`, commitTimestamp: `2026-09-${String(day).padStart(2, '0')}T11:00:00Z`, provenance: { rocjitsuCommitSha: 'abcdef123456' }, tests: [] }));
  const record = { run: runs[1], test: { mode: 'default', status: 'completed', durationSeconds: 0 } };
  captured.model = { runs: [runs[1]], labels: ['Sep 30'], series: [
    { target: 'gfx1250', mode: 'single', name: 'gfx1250 · single', color: '#0f62fe', points: [null], records: [null] },
    { target: 'gfx1250', mode: 'default', name: 'gfx1250 · default', color: '#0f62fe', points: [{ value: 0, record }], records: [record] },
  ] };
  return { data: { runs }, record };
}

test('benchmark lines are dotted for single, solid for default and never estimate unavailable measurements', () => {
  const { data } = setup();
  renderToStaticMarkup(themed(BenchmarkHistoryChart, { data, filters: { targets: ['gfx1250'], suites: ['Triton'], modes: ['single', 'default'] }, benchmark: { id: 'a', name: 'Fictional' }, range: '1W' }));
  const lines = captured.chart.option.series.filter((series) => series.type === 'line');
  expect(lines).toHaveLength(2);
  expect(lines.map((series) => series.lineStyle.type)).toEqual([[0.1, 6], 'solid']);
  // Points are [commit-day x, seconds]; an unavailable result keeps its slot with a null value.
  expect(lines[0].data.every((point) => point.value[1] === null)).toBe(true);
  expect(lines[1].data.map((point) => point.value[1]).filter((value) => value !== null)).toEqual([0]);
  expect(captured.chart.option.series.some((series) => /bridge|Failed or|unavailable/i.test(series.name))).toBe(false);
  expect(lines.every((series) => series.connectNulls === false)).toBe(true);
});

test('benchmark selection replaces its model synchronously before follow-up pointer events', () => {
  const { data } = setup();
  renderToStaticMarkup(themed(BenchmarkHistoryChart, { data, filters: { targets: ['gfx1250'], suites: ['Triton'], modes: ['single', 'default'] }, benchmark: { id: 'a', name: 'Fictional' }, selectedRunIds: ['fictional-30'] }));
  expect(captured.chart.lazyUpdate).toBe(false);
});

test('point selection retains completed zero and ignores unavailable or non-series clicks', () => {
  const { data, record } = setup();
  const onSelectRecord = vi.fn();
  renderToStaticMarkup(themed(BenchmarkHistoryChart, { data, filters: { targets: ['gfx1250'], suites: ['Triton'], modes: ['single', 'default'] }, benchmark: { id: 'a', name: 'Fictional' }, onSelectRecord }));
  captured.chart.onEvents.click({ componentType: 'series', data: { record } });
  expect(onSelectRecord).toHaveBeenCalledWith(record);
  captured.chart.onEvents.click({ componentType: 'series', data: null });
  captured.chart.onEvents.click({ componentType: 'xAxis', data: { record } });
  expect(onSelectRecord).toHaveBeenCalledTimes(1);
});
