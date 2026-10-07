import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import BenchmarkHistoryChart from '../../src/components/benchmarks/BenchmarkHistoryChart.jsx';
const captured = vi.hoisted(() => ({ chart: null, picker: null, data: null, model: null }));
vi.mock('../../src/components/shared/Chart.jsx', () => ({ default: (props) => { captured.chart = props; return createElement('div'); } }));
vi.mock('../../src/components/benchmarks/ChartPointSelector.jsx', () => ({ default: (props) => { captured.picker = props; return createElement('div'); } }));
vi.mock('../../src/data/selectors.js', () => ({ selectBenchmarkSeries: (data) => { captured.data = data; return captured.model; } }));

function setup() {
  const runs = [1, 30].map((day) => ({ runId: `fictional-${day}`, timestamp: `2026-09-${String(day).padStart(2, '0')}T12:00:00Z`, commitTimestamp: `2026-09-${String(day).padStart(2, '0')}T11:00:00Z`, provenance: { rocjitsuCommitSha: 'abcdef123456' }, tests: [] }));
  const record = { run: runs[1], test: { mode: 'MT', status: 'completed', durationSeconds: 0 } };
  captured.model = { runs: [runs[1]], labels: ['Sep 30'], series: [
    { target: 'gfx1250', mode: 'ST', name: 'gfx1250 · ST', color: '#0f62fe', points: [null], records: [null] },
    { target: 'gfx1250', mode: 'MT', name: 'gfx1250 · MT', color: '#0f62fe', points: [{ value: 0, record }], records: [record] },
  ] };
  return { data: { runs }, record };
}

test('benchmark lines use explicit ST solid / MT dashed and never estimate unavailable measurements', () => {
  const { data } = setup();
  renderToStaticMarkup(createElement(BenchmarkHistoryChart, { data, filters: { targets: ['gfx1250'], suites: ['Triton'], modes: ['ST', 'MT'] }, benchmark: { id: 'a', name: 'Fictional' }, range: '1W' }));
  expect(captured.data.runs).toHaveLength(1);
  const lines = captured.chart.option.series.filter((series) => series.type === 'line');
  expect(lines).toHaveLength(2);
  expect(lines.map((series) => series.lineStyle.type)).toEqual(['solid', 'dashed']);
  expect(lines[0].data).toEqual([null]);
  expect(lines[1].data[0].value).toBe(0);
  expect(captured.chart.option.series.some((series) => /bridge|Failed or|unavailable/i.test(series.name))).toBe(false);
  expect(lines.every((series) => series.connectNulls === false)).toBe(true);
});

test('benchmark selection replaces its model synchronously before follow-up pointer events', () => {
  const { data } = setup();
  renderToStaticMarkup(createElement(BenchmarkHistoryChart, { data, filters: { targets: ['gfx1250'], suites: ['Triton'], modes: ['ST', 'MT'] }, benchmark: { id: 'a', name: 'Fictional' }, selectedRunIds: ['fictional-30'] }));
  expect(captured.chart.lazyUpdate).toBe(false);
});

test('point and keyboard result paths retain completed zero and immutable unavailable attempt identity', () => {
  const { data, record } = setup();
  const onSelectRecord = vi.fn(); const onOpenRecord = vi.fn(); const onSelectRun = vi.fn();
  renderToStaticMarkup(createElement(BenchmarkHistoryChart, { data, filters: { targets: ['gfx1250'], suites: ['Triton'], modes: ['ST', 'MT'] }, benchmark: { id: 'a', name: 'Fictional' }, showPointSelector: true, onSelectRecord, onOpenRecord, onSelectRun }));
  captured.chart.onEvents.click({ componentType: 'series', data: { record } });
  expect(onSelectRecord).toHaveBeenCalledWith(record);
  const measured = captured.picker.options.find((choice) => choice.record);
  captured.picker.onActivate(measured);
  expect(onOpenRecord).toHaveBeenCalledWith(record);
  const unavailable = captured.picker.options.find((choice) => !choice.record);
  expect(unavailable.mode).toBe('ST');
  captured.picker.onActivate(unavailable);
  expect(onSelectRun).toHaveBeenCalledWith('fictional-30');
});
