import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { themed } from '../helpers/themed.js';
import { expect, test, vi } from 'vitest';
import BenchmarkHistoryChart from '../../src/components/benchmarks/BenchmarkHistoryChart.jsx';
import { chartPointSize } from '../../src/utils/chartStyles.js';
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
  expect(lines.every((series) => series.connectNulls === true)).toBe(true);
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

// The line connects across empty days, so the dots decide which days really have a result.
function dotSizes(values) {
  const runs = values.map((_, day) => ({ runId: `fictional-${day}`, timestamp: `2026-09-${String(day + 1).padStart(2, '0')}T12:00:00Z`, commitTimestamp: `2026-09-${String(day + 1).padStart(2, '0')}T11:00:00Z`, tests: [] }));
  const records = values.map((value, day) => value === null ? null : { run: runs[day], test: { mode: 'default', status: 'completed', durationSeconds: value } });
  const points = values.map((value, day) => value === null ? null : { value, record: records[day] });
  captured.model = { runs, labels: runs.map((_, day) => `Sep ${day + 1}`), series: [
    { target: 'gfx1250', mode: 'default', name: 'gfx1250 · default', color: '#0f62fe', points, records, measured: points.map((point) => point ?? { value: null }) },
  ] };
  renderToStaticMarkup(themed(BenchmarkHistoryChart, { data: { runs }, filters: { targets: ['gfx1250'], suites: ['Triton'], modes: ['default'] }, benchmark: { id: 'a', name: 'Fictional' }, range: 'ALL' }));
  const line = captured.chart.option.series.find((series) => series.type === 'line');
  return { line, sizes: values.map((_, day) => line.symbolSize(null, { dataIndex: day })) };
}

// A narrow chart thins dots (every 4th here), so these cases decide which measured points keep one.
const measuredAt = (...days) => Array.from({ length: 9 }, (_, day) => ({ value: days.includes(day) ? 1 : null }));
const sizesAt = (points) => points.map((_, index) => chartPointSize(points, index, 'default', 'value', 200, 44));

test('a lone measured point keeps its dot even when thinning would skip it', () => {
  expect(sizesAt(measuredAt(5))).toEqual([0, 0, 0, 0, 0, 7, 0, 0, 0]);
});

test('both ends of a measured stretch keep a dot after leading and trailing empty days', () => {
  const sizes = sizesAt(measuredAt(1, 2, 3, 4, 5, 6, 7));
  expect(sizes[0]).toBe(0);
  expect(sizes[1]).toBe(7);
  expect(sizes[7]).toBe(7);
  expect(sizes[2]).toBe(0);
});

test('the chart keeps connecting the line across empty days', () => {
  const { line } = dotSizes([null, null, 3, null, null]);
  expect(line.connectNulls).toBe(true);
});

test('an empty day plots no value and carries no record, so a click on a bridged stretch selects nothing', () => {
  const { line } = dotSizes([1, null, 3]);
  expect(line.data[1].value[1]).toBeNull();
  expect(line.data[1].record).toBeUndefined();
  const onSelectRecord = vi.fn();
  renderToStaticMarkup(themed(BenchmarkHistoryChart, { data: { runs: [] }, filters: { targets: ['gfx1250'], suites: ['Triton'], modes: ['default'] }, benchmark: { id: 'a', name: 'Fictional' }, range: 'ALL', onSelectRecord }));
  captured.chart.onEvents.click({ componentType: 'series', data: line.data[1] });
  expect(onSelectRecord).not.toHaveBeenCalled();
});
