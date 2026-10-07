import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import MetricsGrid from '../../src/components/overview/MetricsGrid.jsx';
import DurationHistory from '../../src/components/overview/DurationHistory.jsx';
import * as trend from '../../src/components/overview/overviewPresentation.js';
import LargestChanges from '../../src/components/overview/LargestChanges.jsx';
import OverviewView from '../../src/components/overview/OverviewView.jsx';

test('overview passes its selected period into the metric strip', () => {
  const html = render(OverviewView, { data: { runs: [] }, viewModel: { metrics: { total: 0 }, history, changes: [] }, state: { filters: { targets: [], suites: [], modes: [] }, historyRange: '1W', setHistoryRange() {} } });
  expect(html).toContain('Selected period · past week');
});

test('trend explicitly discloses selected configuration and benchmark coverage', () => {
  const html = render(DurationHistory, { history, filters: { targets: ['gfx1250', 'gfx950'], suites: ['Triton'], modes: ['ST', 'MT'] }, coverage: { completed: 12, total: 12 }, range: '1W', onRangeChange() {} });
  expect(html).toContain('2 targets · 1 suite · ST + MT');
  expect(html).toContain('12/12 selected benchmark results');
});

test('largest changes uses exact scope caption, shape tags and relative magnitude bars without old footer', () => {
  const changes = [-10, 5].map((delta, i) => ({ delta, candidateTest: { testId: `gfx1250:ST:w${i}`, name: `Fictional ${i}`, target: 'gfx1250', suite: 'Triton', mode: 'ST' } }));
  const html = render(LargestChanges, { changes });
  expect(html).toContain('Across selected targets and suites and execution modes');
  expect(html).toContain('data-category="mode"');
  expect(html).toContain('data-testid="change-magnitude-bar"');
  expect(html).toContain('width:100%');
  expect(html).toContain('width:50%');
  expect(html).not.toContain('Negative = faster');
});

const chartCapture = vi.hoisted(() => ({ props: null }));
vi.mock('../../src/components/shared/Chart.jsx', () => ({ default: (props) => { chartCapture.props = props; return createElement('div', { role: 'img', 'aria-label': props.ariaLabel }); } }));
const history = {
  series: [{ target: 'Selected runtime', color: '#0f62fe', data: [120, null, 100], baseline: 120 }],
  slots: [0, 1, 2].map((x) => ({ x, run: { runId: `fictional-${x}`, timestamp: `2026-10-0${x + 1}T12:00:00Z`, commitTimestamp: `2026-10-0${x + 1}T11:00:00Z`, provenance: { rocjitsuCommitSha: 'abcdef123456' } } })),
  dayKeys: ['2026-10-01', '2026-10-02', '2026-10-03'], axisMax: 2,
  currentDuration: 100, durationDelta: -16.67, summary: '1 target · 1 suite · ST',
};

test('summed trend has a derived dotted baseline and connects visual gaps without changing null data', () => {
  const html = render(DurationHistory, { history, range: '1M', onRangeChange() {} });
  const option = chartCapture.props.option;
  expect(option.series).toHaveLength(1);
  expect(option.series[0].connectNulls).toBe(true);
  expect(option.series[0].markLine.label.formatter).toBe('Baseline · 2m0s');
  expect(option.series[0].markLine.lineStyle.type).toBe('dotted');
  expect(option.series[0].data[1][1]).toBeNull();
  expect(option.series[0].lineStyle.shadowBlur).toBeGreaterThan(0);
  expect(html).toContain('Sum of selected benchmark runtimes');
  expect(html).toContain('Arrow keys');
  expect(html).toContain('tabindex="0"');
  expect(chartCapture.props.onEvents).toHaveProperty('updateAxisPointer');
  expect(chartCapture.props.onEvents).toHaveProperty('click');
  expect(chartCapture.props.onEvents).toHaveProperty('finished');
});

test('normalized trend tooltip is compact and leaves anchor provenance to inspection details', () => {
  const estimatedHistory = { ...history, normalized: true, series: [{ ...history.series[0], estimated: [true, false, false] }], estimates: [{ estimatedRunId: 'fictional-0', runId: 'fictional-anchor-first-success', target: 'gfx1250', mode: 'ST', logicalTestId: 'added-workload', durationSeconds: 4, timestamp: '2026-10-04T11:00:00Z' }] };
  render(DurationHistory, { history: estimatedHistory, range: '1W', onRangeChange() {} });
  const option = chartCapture.props.option;
  const detail = option.tooltip.formatter([{ dataIndex: 0, value: option.series[0].data[0] }]);
  expect(detail).toContain('Normalized estimate');
  expect(detail).not.toContain('first success');
  expect(detail).not.toContain('fictional-anchor-first-success');
  expect(detail).not.toContain('added-workload');
  expect(detail).not.toContain('fictional-0');
  expect(detail).not.toContain('Execution time');
  expect(option.tooltip.confine).toBe(true);
  expect(option.tooltip.extraCssText).toContain('max-width:240px');
});

test('trend keyboard and nearest inspection navigate eligible points rather than missing slots', () => {
  const indexes = [0, 2, 7];
  expect(trend.trendKeyIndex?.(indexes, null, 'ArrowRight')).toBe(0);
  expect(trend.trendKeyIndex?.(indexes, 0, 'ArrowRight')).toBe(2);
  expect(trend.trendKeyIndex?.(indexes, 2, 'ArrowLeft')).toBe(0);
  expect(trend.trendKeyIndex?.(indexes, 0, 'End')).toBe(7);
  expect(trend.trendKeyIndex?.(indexes, 7, 'Home')).toBe(0);
  expect(trend.trendKeyIndex?.(indexes, 7, 'Escape')).toBeNull();
  expect(trend.nearestTrendIndex?.(history, 1.8)).toBe(2);
  expect(trend.nearestTrendIndex?.(history, 0.2)).toBe(0);
  expect(trend.nearestTrendIndex?.(history, 1)).toBeNull();
});

test('mouse/pen pointer exit hides guides without clearing persistent inspection', () => {
  expect(trend.shouldHideTrendPointer('touch')).toBe(false);
  expect(trend.shouldHideTrendPointer('keyboard')).toBe(false);
  expect(trend.shouldHideTrendPointer('mouse')).toBe(true);
  expect(trend.shouldHideTrendPointer('pen')).toBe(true);
});

const render = (component, props) => renderToStaticMarkup(createElement(component, props));

test('metric strip preserves icon identities and selected-period summed runtime meaning', () => {
  const html = render(MetricsGrid, { range: '1M', metrics: { duration: 120, durationDelta: -8, completed: 2, total: 2, failed: 0 } });
  for (const icon of ['TimerRounded', 'SpeedRounded', 'FactCheckRounded', 'HealthAndSafetyRounded']) expect(html).toContain(`data-testid="${icon}Icon"`);
  expect(html).toContain('Sum of selected benchmark runtimes');
  expect(html).toContain('Selected period · past month');
  expect(html).not.toContain('All history');
  const coverage = html.slice(html.indexOf('data-testid="metric-card-run-coverage"')).split('data-testid="metric-card-run-health"')[0];
  expect(coverage).toContain('data-tone="success"');
  for (const metrics of [{ completed: 0, total: 0 }, { completed: 1, total: 2 }]) {
    const empty = render(MetricsGrid, { range: '1W', metrics: { ...metrics, failed: 0 } });
    const card = empty.slice(empty.indexOf('data-testid="metric-card-run-coverage"')).split('data-testid="metric-card-run-health"')[0];
    expect(card).toContain('data-tone="neutral"');
  }
});
