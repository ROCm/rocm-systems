import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import CompareRunsView from '../../src/components/views/CompareRunsView.jsx';
import RunMetadataDiff from '../../src/components/compare/RunMetadataDiff.jsx';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
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
const filters = { targets: ['gfx1250'], suites: ['Triton'], modes: ['ST'] };
const testResult = { testId: 'gfx1250:ST:a', logicalTestId: 'a', name: 'Fictional GEMM', target: 'gfx1250', mode: 'ST', suite: 'Triton', status: 'completed', durationSeconds: 10 };
const run = (runId, duration, branch = 'develop') => ({ runId, branch, timestamp: '2026-10-05T14:00:00Z', commitTimestamp: '2026-10-05T13:00:00Z', modes: ['ST'], targets: ['gfx1250'], provenance: { rocjitsuCommitSha: runId.padEnd(40, 'a') }, tests: [{ ...testResult, durationSeconds: duration }] });
const baseline = run('fictional-base', 10);
const candidate = run('fictional-next', 8, 'topic/fictional');
const data = { runs: [baseline], allRuns: [baseline, candidate], latestRun: baseline };
const render = (props = {}) => { captured.selectors = []; return renderToStaticMarkup(createElement(CompareRunsView, { data, filters, selectedBaselineId: baseline.runId, selectedCandidateId: candidate.runId, onBaselineChange() {}, onCandidateChange() {}, ...props })); };

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

test.each(['baseline', 'candidate'])('ST-only %s reports unpublished MT before catalog absence', (side) => {
  const source = createSchema2Publication();
  // Different catalogs also exercise genuinely absent ST workloads.
  const stOnly = source.runs[0];
  stOnly.configurations = stOnly.configurations.filter(({ mode }) => mode === 'ST');
  const both = source.runs[8];
  source.runs = [stOnly, both];
  source.index.runFiles = source.runs.map(({ id }) => `runs/${id}.json`);
  const normalized = validatePublishedDashboardData(source).data;
  const props = {
    data: normalized,
    filters: { targets: ['gfx1250'], suites: ['Triton', 'Llama'], modes: ['ST', 'MT'] },
    selectedBaselineId: side === 'baseline' ? stOnly.id : both.id,
    selectedCandidateId: side === 'candidate' ? stOnly.id : both.id,
  };
  const rows = render(props).match(/<tr\b[^>]*>[\s\S]*?<\/tr>/g);
  const cellsFor = (name, mode) => rows.find((row) => row.includes(name) && row.includes(`gfx1250 · ${mode}`))
    .match(/<td\b[^>]*>[\s\S]*?<\/td>/g);
  const missingIndex = side === 'baseline' ? 3 : 4;
  const publishedIndex = side === 'baseline' ? 4 : 3;
  for (const name of ['Fictional GEMM', 'Fictional added workload D']) {
    const cells = cellsFor(name, 'MT');
    expect(cells[missingIndex]).toContain('Configuration not published');
    expect(cells[missingIndex]).not.toContain('Unavailable in catalog');
    expect(cells[publishedIndex]).toContain('Completed');
  }
  expect(cellsFor('Fictional added workload D', 'ST')[missingIndex]).toContain('Unavailable in catalog');
  expect(cellsFor('Fictional added workload D', 'ST')[publishedIndex]).toContain('Failed');
  expect(cellsFor('Fictional added workload E', 'ST')[publishedIndex]).toContain('Timeout');
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
  expect(captured.chart.option.yAxis.data[0]).toContain('ST');
  expect(captured.chart.option.xAxis.axisLabel.fontFamily).toBeTruthy();
});

test('measured zero baseline remains inspectable but has no synthetic percentage bar or no-results claim', () => {
  captured.chart = null;
  const zeroBaseline = { ...baseline, tests: [{ ...testResult, durationSeconds: 0 }] };
  const html = render({ data: { ...data, allRuns: [zeroBaseline, candidate] } });
  expect(html).toContain('aria-label="Comparable benchmark results"');
  expect(html).toContain('percentage unavailable');
  expect(html).toContain('No percentage changes available');
  expect(html).not.toContain('No completed benchmark results are comparable');
  expect(captured.chart).toBeNull();
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
