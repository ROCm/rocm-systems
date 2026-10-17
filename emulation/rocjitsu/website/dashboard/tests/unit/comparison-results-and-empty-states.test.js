import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import CompareRunsView from '../../src/components/views/CompareRunsView.jsx';
import { benchmarkData } from '../fixtures/publishedData.js';
import { selectRunComparison } from '../../src/data/selectors.js';
import { formatDuration, formatPercent } from '../../src/utils/formatters.js';

const filters = { targets: benchmarkData.targets, suites: benchmarkData.suites, modes: benchmarkData.modes };
function renderComparison(baseline, candidate, scope = filters) {
  return renderToStaticMarkup(createElement(CompareRunsView, {
    data: { ...benchmarkData, runs: [baseline, candidate], allRuns: [baseline, candidate], latestRun: candidate },
    filters: scope, selectedBaselineId: baseline.runId, selectedCandidateId: candidate.runId,
    onBaselineChange() {}, onCandidateChange() {},
  }));
}

test('metadata aligns the arbitrary key union baseline-left and candidate-right', () => {
  const source = benchmarkData.runs.at(-1);
  const baseline = { ...source, runId: 'baseline', provenance: { ...source.provenance, details: [{ key: 'onlyBase', label: 'Base tool', value: 'old-tool' }] } };
  const candidate = { ...source, runId: 'candidate', provenance: { ...source.provenance, details: [{ key: 'onlyCandidate', label: 'New tool', value: 'new-tool' }] } };
  const html = renderComparison(baseline, candidate);
  expect(html).toContain('data-testid="metadata-row-onlyBase"');
  expect(html).toContain('data-testid="metadata-row-onlyCandidate"');
  const row = html.slice(html.indexOf('data-testid="metadata-row-onlyBase"')).split('</tr>')[0];
  expect(row.indexOf('old-tool')).toBeLessThan(row.indexOf('Not provided'));
  expect(html.indexOf('Baseline run')).toBeLessThan(html.indexOf('Candidate run'));
});

test('an empty selected scope is not presented as successful comparison or zero-duration totals', () => {
  const html = renderComparison(benchmarkData.runs.at(-2), benchmarkData.runs.at(-1), { targets: [], suites: [], modes: [] });
  expect(html.includes('Every selected benchmark has completed data in both runs.')).toBe(false);
  expect(html.includes('No benchmarks in the selected scope.')).toBe(true);
  expect(html.includes('data-testid="comparison-results"')).toBe(true);
});

test('run selectors expose full attempt identity outside the clipped input', () => {
  const html = renderComparison(benchmarkData.runs.at(-2), benchmarkData.runs.at(-1));
  expect(html.includes('data-testid="baseline-run-selected-identity"')).toBe(true);
  expect(html.includes('data-testid="candidate-run-selected-identity"')).toBe(true);
});

test('a missing baseline is not mislabelled as catalog absence', () => {
  const candidate = benchmarkData.runs[0];
  const html = renderToStaticMarkup(createElement(CompareRunsView, {
    data: { ...benchmarkData, runs: [candidate], allRuns: [candidate], latestRun: candidate }, filters,
    selectedCandidateId: candidate.runId, selectedBaselineId: null,
    onBaselineChange() {}, onCandidateChange() {},
  }));
  expect(html.includes('Unavailable in catalog')).toBe(false);
  expect(html.includes('No run selected')).toBe(true);
  expect(html).toContain('Select baseline and candidate runs to compare benchmark results.');
  expect(html).not.toContain('No completed benchmark results are comparable between these runs.');
});

test('selected run labels retain one clock while metadata excludes bookkeeping clocks', () => {
  const html = renderComparison(benchmarkData.runs.at(-2), benchmarkData.runs.at(-1));
  const inputs = [...html.matchAll(/<input[^>]*role="combobox"[^>]*value="([^"]*)"/g)].map((match) => match[1]);
  expect(inputs).toHaveLength(2);
  expect(inputs.every((value) => value.split('UTC').length === 2)).toBe(true);
  for (const key of ['commitTime', 'runTime']) expect(html).not.toContain(`data-testid="metadata-row-${key}"`);
  for (const key of ['coverage', 'catalog', 'machine', 'configurations']) expect(html).toContain(`data-testid="metadata-row-${key}"`);
});

test('accessible result rows use the shared comparable pairs, sums, deltas and symmetric exclusions', () => {
  const source = benchmarkData.runs.at(-1);
  const test = source.tests[0];
  const baseline = { ...source, runId: 'base', tests: [
    { ...test, durationSeconds: 10 },
    { ...test, testId: 'base-only', name: 'Baseline only', durationSeconds: 999 },
    { ...test, testId: 'failure', name: 'Failed pair', durationSeconds: 40 },
  ] };
  const candidate = { ...source, runId: 'next', tests: [
    { ...test, durationSeconds: 8 },
    { ...test, testId: 'candidate-only', name: 'Candidate only', durationSeconds: 888 },
    { ...test, testId: 'failure', name: 'Failed pair', status: 'timeout', durationSeconds: null },
  ] };
  for (const [base, next] of [[baseline, candidate], [candidate, baseline]]) {
    const model = selectRunComparison(next, base, filters);
    expect(model.comparable).toHaveLength(1);
    expect(model.notComparable).toHaveLength(3);
    const html = renderComparison(base, next);
    expect(html.includes('aria-label="Comparable benchmark results"')).toBe(true);
    expect(html.includes(formatDuration(model.baselineDuration))).toBe(true);
    expect(html.includes(formatDuration(model.candidateDuration))).toBe(true);
    expect(html.includes(formatPercent(model.comparable[0].delta))).toBe(true);
    expect(html.includes('Baseline only')).toBe(true);
    expect(html.includes('Candidate only')).toBe(true);
    expect(html.includes('Timeout')).toBe(true);
  }
});
