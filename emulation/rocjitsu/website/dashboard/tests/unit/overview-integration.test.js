import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import LargestChanges from '../../src/components/overview/LargestChanges.jsx';
import MetricsGrid from '../../src/components/overview/MetricsGrid.jsx';
import DurationHistory from '../../src/components/overview/DurationHistory.jsx';
import { readDashboardRoute, buildDashboardUrl } from '../../src/hooks/useDashboardState.js';
import { formatFullDate } from '../../src/utils/formatters.js';
import { trendSlotStatus } from '../../src/components/overview/overviewPresentation.js';

const capture = vi.hoisted(() => ({ props: null }));
vi.mock('../../src/components/shared/Chart.jsx', () => ({ default: (props) => { capture.props = props; return createElement('div'); } }));
const run = { runId: 'attempt', timestamp: '2026-01-01T12:00:00Z', provenance: { rocjitsuCommitSha: 'abc' } };
const point = { run, runId: 'attempt', durationSeconds: 12, speedup: 1, required: 2, completed: 2, reason: null };
const projection = { rows: [{ key: 'gfx950/single', target: 'gfx950', mode: 'single', cohortIds: ['a', 'b'], comparedWorkloads: 2, reference: point, latest: point, points: [point, { ...point, run: null, speedup: null, durationSeconds: null }] }],
  slots: [{ run, x: 0 }, { run: null, x: 1 }], range: '1M', dayKeys: ['2026-01-01', '2026-01-02'], axisMax: 1, endpointLabel: 'Latest selected run' };

test('all-gap comparisons retain a focusable chronological surface without a chart', () => {
  const gap = { ...point, durationSeconds: null, speedup: null, completed: 0, reason: 'timeout' };
  const allGap = { ...projection, attempts: [{ run, eligible: false }], rows: [{ ...projection.rows[0], points: [gap, { ...gap, run: null }], reference: null, latest: gap }] };
  const html = renderToStaticMarkup(createElement(DurationHistory, { projection: allGap, catalogs: [], onRangeChange() {}, onCatalogChange() {} }));
  expect(html).toMatch(/data-testid="performance-trend-chart"[^>]*tabindex="0"/);
  expect(html).toContain('data-testid="trend-inspection-status"');
  expect(html).toContain('aria-live="polite"');
  expect(html).not.toContain('data-testid="point-hover-guide"');
});

test('cards consume per-configuration endpoints and show only succeeded/required', () => {
  const html = renderToStaticMarkup(createElement(MetricsGrid, { projection }));
  const cards = [...html.matchAll(/data-testid="metric-card-([^"]+)"/g)].map((match) => match[1]);
  expect(cards).toEqual(['latest-duration', 'relative-speedup', 'latest-results', 'valid-data-points']);
  expect(html).not.toContain('Compared workloads');
  expect(html).toContain('2/2');
  expect(html).toContain('1.00×');
  expect(html).not.toContain('Run health');
});
test('relative chart has only configuration series, no gap bridges, dots or fill, and gap tooltip is empty', () => {
  renderToStaticMarkup(createElement(DurationHistory, { projection, catalogs: [], onRangeChange() {}, onCatalogChange() {} }));
  const option = capture.props.option;
  expect(option.series).toHaveLength(1);
  expect(option.series[0]).toMatchObject({ connectNulls: false, showSymbol: false, symbol: 'none' });
  expect(option.series[0].areaStyle).toBeUndefined();
  expect(option.tooltip.hideDelay).toBe(0);
  expect(option.tooltip.formatter([{ dataIndex: 1 }])).toBe('');
  expect(option.tooltip.formatter([{ dataIndex: 0 }])).toContain('Commit abc');
});
test.each([
  { duration: 12.3456, reference: 60.9876, expected: '12.3s', expectedReference: '1m 1.0s' },
  { duration: 12, reference: 0.1, expected: '12.0s', expectedReference: '0.10s' },
  { duration: 0.004, reference: 119.999, expected: '0.00s', expectedReference: '2m 0.0s' },
  { duration: 75.25, reference: 9.876, expected: '1m 15.3s', expectedReference: '9.88s' },
])('trend tooltip and keyboard status share duration units and precision: $duration', ({ duration, reference, expected, expectedReference }) => {
  const row = projection.rows[0];
  const model = { ...projection, rows: [{ ...row,
    points: [{ ...point, durationSeconds: duration }, row.points[1]],
    reference: { ...point, durationSeconds: reference },
  }] };
  renderToStaticMarkup(createElement(DurationHistory, { projection: model, catalogs: [], onRangeChange() {}, onCatalogChange() {} }));
  const html = capture.props.option.tooltip.formatter([{ dataIndex: 0 }]);
  expect(html).toContain(`<b>1.000×</b> · ${expected} · ref ${expectedReference}`);
  expect(trendSlotStatus(model, 0)).toContain(`gfx950 single: 1.000× · ${expected}`);
});

test('trend tooltip starts with commit identity and commit time rather than execution time', () => {
  const committed = { ...run, commitTimestamp: '2025-12-31T23:00:00Z' };
  renderToStaticMarkup(createElement(DurationHistory, { projection: { ...projection, slots: [{ run: committed, x: 0 }, projection.slots[1]] }, catalogs: [], onRangeChange() {}, onCatalogChange() {} }));
  const html = capture.props.option.tooltip.formatter([{ dataIndex: 0 }]);
  expect(html.split('<br/>').slice(0, 2)).toEqual(['<b>Commit abc</b>', formatFullDate(committed.commitTimestamp)]);
  expect(html).not.toContain(formatFullDate(run.timestamp));
  expect(html).not.toContain('Attempt:');
  expect(html).not.toContain('Reference:');
  expect(html.split('<br/>').slice(2)).toEqual(['gfx950 single · <b>1.000×</b> · 12.0s · ref 12.0s']);
});

test.each(['1M', '3M', 'ALL'])('%s date labels use regular whole-day intervals without short edge ticks', (range) => {
  const model = { ...projection, range, axisMax: 23,
    dayKeys: Array.from({ length: 24 }, (_, index) => new Date(Date.UTC(2026, 0, index + 1)).toISOString().slice(0, 10)),
    slots: [{ run: null, x: 0 }, { run, x: 1 }, { run, x: 23 }],
    rows: [{ ...projection.rows[0], points: [{ ...point, speedup: null }, point, { ...point, speedup: 1.04 }] }],
  };
  renderToStaticMarkup(createElement(DurationHistory, { projection: model, catalogs: [], onRangeChange() {}, onCatalogChange() {} }));
  const axis = capture.props.option.xAxis;
  expect(axis.axisLabel.customValues).toEqual(range === 'ALL' ? [1, 5, 9, 13, 17, 21] : [0, 4, 8, 12, 16, 20]);
  expect(axis.axisLabel.customValues.map(axis.axisLabel.formatter)).toEqual(range === 'ALL'
    ? ['Jan 2', 'Jan 6', 'Jan 10', 'Jan 14', 'Jan 18', 'Jan 22']
    : ['Jan 1', 'Jan 5', 'Jan 9', 'Jan 13', 'Jan 17', 'Jan 21']);
  expect(axis.max).toBe(23);
});

test('trend uses a solid bottom axis and a dotted reference only for sub-baseline results', () => {
  const render = (model) => renderToStaticMarkup(createElement(DurationHistory, { projection: model, catalogs: [], onRangeChange() {}, onCatalogChange() {} }));
  render(projection);
  expect(capture.props.option.xAxis.axisLine).toMatchObject({ show: true, onZero: false, lineStyle: { type: 'solid' } });
  expect(capture.props.option.series[0].markLine).toBeUndefined();
  render({ ...projection, rows: [{ ...projection.rows[0], points: [point, { ...point, speedup: 0.96 }] }] });
  expect(capture.props.option.series[0].markLine).toMatchObject({ lineStyle: { type: 'dotted' }, data: [{ yAxis: 1 }] });
});

test.each(['1W', '1M', '3M', 'ALL'])('%s preserves its selected start policy and interior gaps', (range) => {
  const start = range === '1W' ? 1.5 : 5;
  const model = { ...projection, range, axisMax: 10,
    slots: [{ run: null, x: 0 }, { run, x: start }, { run: null, x: start + 1 }, { run, x: start + 2 }],
    rows: [{ ...projection.rows[0], points: [{ ...point, speedup: null }, point, { ...point, speedup: null }, { ...point, speedup: 1.04 }] }],
  };
  renderToStaticMarkup(createElement(DurationHistory, { projection: model, catalogs: [], onRangeChange() {}, onCatalogChange() {} }));
  expect(capture.props.option.xAxis.min).toBe(['1M', '3M'].includes(range) ? 0 : start);
  expect(capture.props.option.series[0].data[2][1]).toBeNull();
});

test('weekly axes align day-center labels and the first baseline point with the plot origin', () => {
  const weekly = { ...projection, range: '1W', axisMax: 3,
    dayKeys: ['2026-01-01', '2026-01-02', '2026-01-03'],
    slots: [{ run, x: 0.5 }, { run: null, x: 1 }, { run, x: 2.5 }],
    rows: [{ ...projection.rows[0], points: [point, { ...point, speedup: null }, { ...point, speedup: 1.04 }] }],
  };
  const render = (model) => renderToStaticMarkup(createElement(DurationHistory, { projection: model, catalogs: [], onRangeChange() {}, onCatalogChange() {} }));
  render(weekly);
  const { xAxis, yAxis, series } = capture.props.option;
  expect(xAxis.min).toBe(0.5);
  expect(xAxis.max).toBe(2.5);
  expect(xAxis.axisLabel.customValues).toEqual([0.5, 1.5, 2.5]);
  expect(xAxis.axisLabel.customValues.map(xAxis.axisLabel.formatter)).toEqual(['Jan 1', 'Jan 2', 'Jan 3']);
  expect(series[0].data).toEqual([[0.5, 1], [1, null], [2.5, 1.04]]);
  expect(yAxis.min({ min: 1 })).toBe(1);
  expect(yAxis.min({ min: 0.96 })).toBeLessThan(0.96);
  render({ ...weekly, slots: [{ run: null, x: 0 }, ...weekly.slots.slice(1)] });
  expect(capture.props.option.xAxis.min).toBe(2.5);
  render({ ...weekly, slots: [{ run, x: 0.25 }, { run, x: 0.75 }, { run, x: 2.75 }] });
  expect(capture.props.option.xAxis.min).toBe(0.25);
  expect(capture.props.option.xAxis.max).toBe(2.75);
});

test('Largest Changes ranks one flat six-entry list across configurations with original item labels', () => {
  const row = (target, deltas) => ({ ...projection.rows[0], key: `${target}/single`, target,
    reference: { ...point, results: deltas.map((_, i) => ({ id: `work-${i}`, result: { durationSeconds: 100 } })) },
    latest: { ...point, results: deltas.map((delta, i) => ({ id: `work-${i}`, result: { name: `${target}-${i}`, suite: 'Triton', durationSeconds: 100 + delta } })) } });
  const html = renderToStaticMarkup(createElement(LargestChanges, { projection: { ...projection, catalogId: 'current', rows: [row('gfx950', [10, 30, 50, 70]), row('gfx1250', [-20, -40, -60, -80])] } }));
  expect((html.match(/data-change-state=/g) ?? [])).toHaveLength(6);
  const names = ['gfx1250-3', 'gfx950-3', 'gfx1250-2', 'gfx950-2', 'gfx1250-1', 'gfx950-1'];
  expect(names.map((name) => html.indexOf(`>${name}<`))).toEqual([...names.map((name) => html.indexOf(`>${name}<`))].sort((a, b) => a - b));
  expect(html).not.toContain('>gfx950-0<');
  expect(html).not.toContain('data-configuration-group');
  expect(html).not.toContain('combobox');
  for (const label of ['gfx950', 'gfx1250', 'Triton', 'single']) expect(html).toContain(`>${label}<`);
});

test('Largest Changes applies displayed-catalog shared membership before ranking and limit', () => {
  const catalogs = [
    { catalog: { id: 'old', configurations: { 'gfx950/single': ['a', 'b', 'retired'] } } },
    { catalog: { id: 'current', configurations: { 'gfx950/single': ['a', 'b', 'added'] } } },
  ];
  const ids = ['a', 'b', 'added'];
  const row = { ...projection.rows[0], cohortIds: ids,
    reference: { ...point, results: ids.map((id) => ({ id, result: { durationSeconds: 100 } })) },
    latest: { ...point, results: ids.map((id, i) => ({ id, result: { name: id, suite: 'Triton', durationSeconds: i === 2 ? 1000 : 110 + i } })) } };
  const render = (catalogId) => renderToStaticMarkup(createElement(LargestChanges, { catalogs, projection: { ...projection, catalogId, rows: [row] } }));
  const shared = render(null);
  expect((shared.match(/data-change-state=/g) ?? [])).toHaveLength(2);
  expect(shared).not.toContain('data-workload="added"');
  expect(render('current')).toContain('data-workload="added"');
});

test('Largest Changes excludes invalid endpoints and preserves empty/unavailable reasons', () => {
  const unavailable = { ...projection.rows[0], latest: { ...point, reason: 'failed', durationSeconds: null } };
  const render = (rows) => renderToStaticMarkup(createElement(LargestChanges, { projection: { ...projection, catalogId: 'current', rows } }));
  expect(render([unavailable])).toContain('failed');
  expect(render([unavailable])).not.toContain('data-change-state');
  expect(render([])).toContain('No configurations selected');
});

test('Largest Changes keeps first duplicate reference, strict special-string IDs, ties and inputs', () => {
  const ids = ['__proto__', 'constructor', '1', 'missing'];
  const first = { durationSeconds: 100 };
  const row = { ...projection.rows[0], cohortIds: ids,
    reference: { ...point, results: [
      { id: '__proto__', result: first }, { id: '__proto__', result: { durationSeconds: 1 } },
      { id: 'constructor', result: { durationSeconds: 100 } },
      { id: 1, result: { durationSeconds: 1 } }, { id: '1', result: { durationSeconds: 100 } },
    ] },
    latest: { ...point, results: ids.map((id) => ({ id, result: { name: `name-${id}`, suite: 'Triton', durationSeconds: 110 } })) },
  };
  const before = structuredClone(row);
  const html = renderToStaticMarkup(createElement(LargestChanges, { projection: { ...projection, catalogId: 'current', rows: [row] } }));
  expect([...html.matchAll(/data-workload="([^"]+)"/g)].map((match) => match[1])).toEqual(['__proto__', '1', 'constructor'].sort((a, b) => a.localeCompare(b)));
  expect((html.match(/\+10\.0%/g) ?? [])).toHaveLength(3);
  expect(row).toEqual(before);
  expect(row.reference.results[0].result).toBe(first);
  const empty = { ...row, latest: { ...row.latest, results: [] } };
  expect(renderToStaticMarkup(createElement(LargestChanges, { projection: { ...projection, catalogId: 'current', rows: [empty] } }))).toContain('No comparable results');
  const unusedReference = { ...row, reference: { ...row.reference, results: undefined } };
  const filtered = { ...projection, catalogId: null, rows: [unusedReference] };
  expect(renderToStaticMarkup(createElement(LargestChanges, { projection: filtered, catalogs: [] }))).toContain('No shared workloads');
});

test('catalog routing preserves unrelated query/hash and explicit selections', () => {
  expect(readDashboardRoute('https://example.test/').historyRange).toBe('ALL');
  const route = readDashboardRoute('https://example.test/?catalog=cat-a&range=1M&targets=&view=overview&keep=yes#anchor');
  expect(route.overviewCatalogId).toBe('cat-a');
  expect(route.preferences.targets).toEqual([]);
  const url = buildDashboardUrl('https://example.test/?keep=yes#anchor', route);
  expect(new URL(url).searchParams.get('catalog')).toBe('cat-a');
  expect(new URL(url).searchParams.get('keep')).toBe('yes');
  expect(new URL(url).hash).toBe('#anchor');
});
