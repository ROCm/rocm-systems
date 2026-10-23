import { fixtureRunPath } from '../fixtures/runPath.js';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import CompareRunsView from '../../src/components/views/CompareRunsView.jsx';
import RunMetadataDiff from '../../src/components/compare/RunMetadataDiff.jsx';
import ComparisonMetrics from '../../src/components/compare/ComparisonMetrics.jsx';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { formatDuration, formatPercent } from '../../src/utils/formatters.js';
import { selectRunComparison } from '../../src/data/selectors.js';
import { colorTokens } from '../../src/theme/tokens.js';

test.each(['light', 'dark'])('%s baseline metric reads purple from the shared palette', (mode) => {
  const original = colorTokens[mode].purpleText;
  const theme = createTheme({ palette: { mode } });
  const model = { comparable: [{}], notComparable: [], candidateDuration: 8, baselineDuration: 10, aggregateDelta: -20 };
  const draw = () => renderToStaticMarkup(createElement(ThemeProvider, { theme }, createElement(ComparisonMetrics, { model })));
  expect(draw()).toContain(`color:${original}`);
  try {
    colorTokens[mode].purpleText = '#123456';
    expect(draw()).toContain('color:#123456');
  } finally {
    colorTokens[mode].purpleText = original;
  }
});
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { setFixtureConfigurations } from '../fixtures/configurationScope.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
const captured = vi.hoisted(() => ({ chart: null, selectors: [], swap: null, pair: null }));
vi.mock('../../src/components/shared/Chart.jsx', () => ({ default: (props) => { captured.chart = props; return createElement('div'); } }));
vi.mock('../../src/components/compare/RunSelector.jsx', () => ({ default: (props) => { captured.selectors.push(props); return createElement('div', null, props.value?.runId ?? 'No run selected'); } }));
vi.mock('@mui/material', async (importOriginal) => {
  const original = await importOriginal();
  return { ...original, Button: (props) => { if (props.children === 'Swap') captured.swap = props; return createElement(original.Button, props); } };
});
vi.mock('../../src/data/selectors.js', async (importOriginal) => {
  const original = await importOriginal();
  return { ...original, selectRunComparison: (...args) => { captured.pair = args; return original.selectRunComparison(...args); } };
});
const filters = { targets: ['gfx1250'], suites: ['Triton'], modes: ['single'] };
const testResult = { testId: 'gfx1250/single:a', logicalTestId: 'a', name: 'Fictional GEMM', target: 'gfx1250', mode: 'single', suite: 'Triton', status: 'completed', durationSeconds: 10 };
const run = (runId, duration, branch = 'develop') => ({ runId, branch, timestamp: '2026-10-05T14:00:00Z', commitTimestamp: '2026-10-05T13:00:00Z', modes: ['single'], targets: ['gfx1250'], provenance: { rocjitsuCommitSha: runId.padEnd(40, 'a') }, tests: [{ ...testResult, durationSeconds: duration }] });
const baseline = run('fictional-base', 10);
const candidate = run('fictional-next', 8, 'topic/fictional');
const data = { runs: [baseline], allRuns: [baseline, candidate], latestRun: baseline };
const render = (props = {}) => { captured.chart = null; captured.selectors = []; return renderToStaticMarkup(createElement(CompareRunsView, { data, filters, selectedBaselineId: baseline.runId, selectedCandidateId: candidate.runId, onBaselineChange() {}, onCandidateChange() {}, ...props })); };

test('comparison uses all published attempts with explicit pair identity and no missing-selection fallback', () => {
  render();
  expect(captured.pair[0]).toBe(candidate);
  expect(captured.pair[1]).toBe(baseline);
  expect(captured.selectors.every((selector) => selector.options.length === 2)).toBe(true);
  const html = render({ selectedCandidateId: 'no-longer-published' });
  expect(captured.pair[0]).toBeNull();
  expect(html).toContain('no-longer-published');
  expect(html).toContain('Selected candidate is unavailable');
  render({ selectedBaselineId: null });
  expect(captured.pair[1]).toBeNull();
});

test('comparison metrics use five real badge icons with signed delta, neutral empty coverage and amber exclusions', () => {
  const html = render();
  for (const icon of ['SpeedRounded', 'TimerRounded', 'HistoryRounded', 'FactCheckRounded', 'FilterAltOffRounded']) expect(html).toContain(`data-testid="${icon}Icon"`);
  expect(html).toContain('-20.0%');
  expect(html).toContain('data-testid="comparison-metric-comparable" data-tone="success"');
  expect(html).toContain('data-testid="comparison-metric-not-comparable" data-tone="neutral"');
  const empty = render({ filters: { targets: [], suites: [], modes: [] } });
  expect(empty).toContain('data-testid="comparison-metric-comparable" data-tone="neutral"');
  const excluded = render({ data: { ...data, allRuns: [baseline, { ...candidate, tests: [{ ...testResult, status: 'failed', durationSeconds: null }] }] } });
  expect(excluded).toContain('data-testid="comparison-metric-not-comparable" data-tone="warning"');
  expect(excluded).not.toContain('Every selected benchmark has completed data');
});

test.each(['baseline', 'candidate'])('single-only %s reports unpublished default before catalog absence', (side) => {
  const source = createMaintainedPublication();
  // Different catalogs also exercise genuinely absent single workloads.
  const stOnly = source.runs[0];
  setFixtureConfigurations(source, stOnly, stOnly.benchmark_results.filter(({ thread_policy }) => thread_policy === 'single'));
  const both = source.runs[8];
  source.runs = [stOnly, both];
  source.index.runFiles = source.runs.map(fixtureRunPath);
  const normalized = adaptPublishedDashboardData(source).data;
  const props = {
    data: normalized,
    filters: { targets: ['gfx1250'], suites: ['Triton', 'Llama'], modes: ['single', 'default'] },
    selectedBaselineId: side === 'baseline' ? stOnly.publication.id : both.publication.id,
    selectedCandidateId: side === 'candidate' ? stOnly.publication.id : both.publication.id,
  };
  const rows = render(props).match(/<tr\b[^>]*>[\s\S]*?<\/tr>/g);
  const cellsFor = (name, mode) => rows.find((row) => row.includes(name) && row.includes(`gfx1250 · ${mode}`))
    .match(/<td\b[^>]*>[\s\S]*?<\/td>/g);
  const missingIndex = side === 'baseline' ? 3 : 4;
  const publishedIndex = side === 'baseline' ? 4 : 3;
  for (const name of ['Fictional GEMM', 'Fictional added workload D']) {
    const cells = cellsFor(name, 'default');
    expect(cells[missingIndex]).toContain('Configuration not published');
    expect(cells[missingIndex]).not.toContain('Unavailable in catalog');
    expect(cells[publishedIndex]).toContain('Completed');
  }
  expect(cellsFor('Fictional added workload D', 'single')[missingIndex]).toContain('Unavailable in catalog');
  expect(cellsFor('Fictional added workload D', 'single')[publishedIndex]).toContain('Failed');
  expect(cellsFor('Fictional added workload E', 'single')[publishedIndex]).toContain('Timeout');
  const noRun = render({ ...props, [side === 'baseline' ? 'selectedBaselineId' : 'selectedCandidateId']: null });
  expect(noRun).toContain('No run selected');
  expect(noRun).not.toContain('Configuration not published');
  expect(noRun).not.toContain('Unavailable in catalog');
});

test('swap invokes an atomic explicit pair callback and benchmark chart options disclose mode and theme fonts', () => {
  const onSwap = vi.fn();
  render({ onSwap });
  captured.swap.onClick();
  expect(onSwap).toHaveBeenCalledExactlyOnceWith({ baselineId: candidate.runId, candidateId: baseline.runId });
  expect(captured.chart.option.textStyle.fontFamily).toBeTruthy();
  expect(captured.chart.option.yAxis.data[0]).toContain('single');
  expect(captured.chart.option.xAxis.axisLabel.fontFamily).toBeTruthy();
});

test('measured zero baseline remains inspectable but has no synthetic percentage bar or no-results claim', () => {
  captured.chart = null;
  const zeroBaseline = { ...baseline, tests: [{ ...testResult, durationSeconds: 0 }] };
  const html = render({ data: { ...data, allRuns: [zeroBaseline, candidate] } });
  expect(html).toContain('aria-label="Comparable benchmark results"');
  expect(html).toContain('percentage unavailable');
  expect(html).toContain('No percentage changes available');
  expect(html).toContain('>0.00s<');
  expect(html).not.toContain('No completed benchmark results are comparable');
  expect(captured.chart).toBeNull();
});

test.each([
  { baselineSeconds: 2.5e-291, delta: 1.6e308, limit: Infinity },
  { baselineSeconds: 5e-291, delta: 8e307, limit: 1e308 },
])('validated $baselineSeconds baseline retains all finite results when the chart domain is unrepresentable', ({ baselineSeconds, delta, limit }) => {
  const source = createMaintainedPublication();
  for (const [id, seconds] of [['fictional-develop-14', baselineSeconds], ['fictional-branch-01', 4e15]]) {
    for (const result of source.runs.find((item) => item.publication.id === id).benchmark_results) {
       Object.assign(result, { status: 'completed', error: null, timing_results_s: [seconds] });
    }
  }
  const normalized = adaptPublishedDashboardData(source).data;
  const props = { data: normalized, filters: { ...filters, suites: ['Triton', 'Llama'] }, selectedBaselineId: 'fictional-develop-14', selectedCandidateId: 'fictional-branch-01' };
  const model = selectRunComparison(normalized.allRuns.find((r) => r.runId === props.selectedCandidateId), normalized.allRuns.find((r) => r.runId === props.selectedBaselineId), props.filters);
  expect(model.comparable).toHaveLength(4);
  expect(model.comparable.every((row) => row.delta === delta)).toBe(true);
  expect(Math.max(5, Math.ceil(delta * 1.25))).toBe(limit);
  expect(Number.isFinite(limit * 2)).toBe(false);
  const html = render(props);
  expect(captured.chart).toBeNull();
  expect(html).toContain('Chart unavailable: percentage changes exceed the supported chart range. Measured results remain in the results table below.');
  expect(html).toContain('View comparable results (4)');
  const rows = html.match(/<tr\b[^>]*>[\s\S]*?<\/tr>/g).filter((row) => row.includes(formatPercent(delta)));
  expect(rows).toHaveLength(4);
  for (const row of rows) {
    expect(row).toContain(formatDuration(baselineSeconds));
    expect(row).toContain(formatDuration(4e15));
  }
  render({ ...props, selectedBaselineId: props.selectedCandidateId, selectedCandidateId: props.selectedBaselineId });
  expect(captured.chart.option.xAxis.min).toBe(-125);
  expect(captured.chart.option.xAxis.max).toBe(125);
  expect(captured.chart.option.series[0].data.every((point) => point.value === -100)).toBe(true);
});

test.each([
  { baselineSeconds: 1e-289, candidateSeconds: 4e15, expectedLimit: 4.999999999999999e306 },
  { baselineSeconds: 10, candidateSeconds: 8, expectedLimit: 25 },
  { baselineSeconds: 10, candidateSeconds: 10, expectedLimit: 5 },
])('validated representable domain $baselineSeconds/$candidateSeconds preserves exact linear bounds', ({ baselineSeconds, candidateSeconds, expectedLimit }) => {
  const source = createMaintainedPublication();
  for (const [id, seconds] of [['fictional-develop-14', baselineSeconds], ['fictional-branch-01', candidateSeconds]]) {
    for (const result of source.runs.find((item) => item.publication.id === id).benchmark_results) {
       Object.assign(result, { status: 'completed', error: null, timing_results_s: [seconds] });
    }
  }
  const normalized = adaptPublishedDashboardData(source).data;
  render({ data: normalized, selectedBaselineId: 'fictional-develop-14', selectedCandidateId: 'fictional-branch-01' });
  const { xAxis, series } = captured.chart.option;
  expect(xAxis.min).toBe(-expectedLimit);
  expect(xAxis.max).toBe(expectedLimit);
  expect(Number.isFinite(xAxis.max - xAxis.min)).toBe(true);
  expect(xAxis.max).toBeGreaterThan(xAxis.min);
  for (const point of series[0].data) {
    expect(point.value).toBeGreaterThanOrEqual(xAxis.min);
    expect(point.value).toBeLessThanOrEqual(xAxis.max);
  }
});

test('validated positive ratio overflow uses generic no-ratio copy and keeps measured rows without a Chart', () => {
  const source = createMaintainedPublication();
  for (const [id, seconds] of [['fictional-develop-14', 1e-300], ['fictional-branch-01', 4e15]]) {
    for (const result of source.runs.find((item) => item.publication.id === id).benchmark_results) {
       Object.assign(result, { status: 'completed', error: null, timing_results_s: [seconds] });
    }
  }
  const normalized = adaptPublishedDashboardData(source).data;
  const html = render({ data: normalized, selectedBaselineId: 'fictional-develop-14', selectedCandidateId: 'fictional-branch-01' });
  expect(captured.chart).toBeNull();
  expect(html).toContain('No percentage changes available. Measured results remain in the results table below.');
  expect(html).not.toContain('Measured zero baselines');
  expect(html).toContain('View comparable results (2)');
  expect(html).toContain(formatDuration(4e15));
  expect(html).toContain(formatDuration(1e-300));
});

test('metadata is a bounded neutral field/baseline/candidate union and preserves full literal source/environment values', () => {
  const base = { ...baseline, sourceBase: { branch: 'develop', commit: 'b'.repeat(40) }, environment: [{ key: 'baseOnly', label: 'Base only', value: '-old+literal' }, { key: 'flag', label: 'Flag', value: false }] };
  const next = { ...candidate, pullRequest: { number: 42, url: 'https://github.com/ROCm/rocm-systems/pull/42' }, environment: [{ key: 'nextOnly', label: 'Next only', value: 'full-value' }, { key: 'flag', label: 'Flag', value: true }] };
  const html = renderToStaticMarkup(createElement(RunMetadataDiff, { baseline: base, candidate: next, filters }));
  expect(html).toContain('>Field<');
  expect(html).toContain('data-testid="metadata-row-baseOnly"');
  expect(html).toContain('data-testid="metadata-row-nextOnly"');
  expect(html).toContain('-old+literal');
  expect(html).not.toContain('data-testid="metadata-row-base"');
  expect(html).not.toContain('Changed');
  expect(html).toContain('data-different="true"');
  expect(html).toContain('Highlighted rows indicate metadata differences, not performance.');
  expect(html).toContain('height:366px');
  expect(html).toContain('position:sticky');
  expect(html).toContain('tabindex="0"');
  expect(html).not.toContain('− original');
  expect(html).not.toContain('Modified value');
});
