import { expect, test } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import ConfigurationMatrix from '../../src/components/branch/ConfigurationMatrix.jsx';
import BenchmarkDifferences from '../../src/components/branch/BenchmarkDifferences.jsx';
import { selectConfigurationComparison } from '../../src/data/branchSelectors.js';
import { configurationState } from '../../src/components/branch/branchPresentation.js';
import { formatDuration } from '../../src/utils/formatters.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { selectRunComparison } from '../../src/data/selectors.js';
import { rowChanges } from '../../src/components/branch/branchPresentation.js';

const filters = { targets: ['gfx1250'], modes: ['single'], suites: ['Triton', 'Llama'] };

test('valid positive wire durations retain measurements when derived percentages overflow', () => {
  const publication = createMaintainedPublication();
  for (const [id, duration] of [['fictional-develop-14', 1e-300], ['fictional-branch-01', 4e15]]) {
    for (const result of publication.runs.find((r) => r.publication.id === id).benchmark_results) {
       Object.assign(result, { status: 'completed', error: null, timing_results_s: [duration] });
    }
  }
  const data = adaptPublishedDashboardData(publication).data;
  const baseline = data.runs.find((r) => r.runId === 'fictional-develop-14');
  const candidate = data.allRuns.find((r) => r.runId === 'fictional-branch-01');
  const model = selectRunComparison(candidate, baseline, filters);
  expect(model.comparable).toHaveLength(4);
  expect(model.counts).toEqual({ faster: 0, slower: 0, neutral: 0, unavailable: 4 });
  expect(model.aggregateDelta).toBeNull();
  for (const row of model.comparable) {
    expect(row.comparable).toBe(true);
    expect(row.delta).toBeNull();
    expect(row.candidateTest.durationSeconds).toBe(4e15);
    expect(row.baselineTest.durationSeconds).toBe(1e-300);
    expect(Number.isFinite(row.deltaSeconds)).toBe(true);
    expect(rowChanges(row)).toEqual({ seconds: row.deltaSeconds, percent: null });
  }
  const reverse = selectRunComparison(baseline, candidate, filters);
  const selection = { target: 'gfx1250', mode: 'single' };
  const comparison = selectConfigurationComparison(candidate, baseline, { ...selection, suites: filters.suites });
  const summary = configurationState(candidate, baseline, comparison, selection.target, selection.mode);
  expect(summary.matched).toBe(4);
  const matrix = renderToStaticMarkup(createElement(ConfigurationMatrix, { candidate, reference: baseline, selection, suites: filters.suites, onSelect() {} }));
  const differences = renderToStaticMarkup(createElement(BenchmarkDifferences, { comparison, summary, candidate, reference: baseline, selection, suites: filters.suites, suiteOptions: filters.suites, query: '', onOpenComparison() {} }));
  for (const html of [matrix, differences]) {
    expect(html).toContain('Percentage unavailable');
    expect(html).not.toContain('zero reference runtime');
  }
  expect(matrix).toContain(formatDuration(comparison.candidateDuration - comparison.baselineDuration));
  expect(differences).toContain(formatDuration(4e15));
  expect(differences).toContain(formatDuration(1e-300));
  expect(differences).toContain('4 matched · 0 excluded');
  expect(differences.match(/data-testid="branch-result-/g)).toHaveLength(4);
  expect(differences.match(/data-testid="branch-mobile-result-/g)).toHaveLength(4);
  expect(differences).toContain('more time');
  expect(reverse.aggregateDelta).toBe(-100);
  expect(reverse.comparable.every((row) => row.delta === -100 && rowChanges(row).percent === -100)).toBe(true);
});

test('zero remains a standalone normalized measurement, not an accepted wire sample', () => {
  const publication = createMaintainedPublication();
  publication.runs[0].benchmark_results[0].timing_results_s = [0];
  expect(adaptPublishedDashboardData(publication).data.runs[0].tests[0].durationSeconds).toBeNull();
  const row = { baselineTest: { durationSeconds: 0 }, candidateTest: { durationSeconds: 2 } };
  expect(rowChanges(row)).toEqual({ seconds: 2, percent: null });
});
