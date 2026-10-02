import { expect, test } from 'vitest';
import { loadDashboardData, validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { selectPluginComparisonGroups } from '../../src/data/pluginComparison.js';
import {
  selectComparisonRuns, selectFailures, selectOverview, selectRecentRuns, selectBenchmarkSeries,
} from '../../src/data/selectors.js';
import { benchmarkData, dataMetadata, publishedCatalogs, publishedRuns } from '../fixtures/publishedData.js';

const filters = { targets: ['gfx1250'], suites: benchmarkData.suites };

function withManualRun(branch, sameSha = false) {
  const source = benchmarkData.latestRun;
  const manual = {
    ...structuredClone(source), runId: 'manual-experiment', comparisonId: 'manual-experiment',
    branch, trigger: 'manual', timestamp: '2026-09-02T12:00:00.000Z',
    commitTimestamp: sameSha ? source.commitTimestamp : '2026-09-02T11:00:00.000Z',
    provenance: { ...source.provenance, rocjitsuCommitSha: sameSha ? source.provenance.rocjitsuCommitSha : 'f'.repeat(40) },
  };
  manual.tests[0] = { ...manual.tests[0], durationSeconds: null, status: 'failed', error: 'Experiment failed' };
  manual.tests.push({ ...manual.tests[1], testId: 'experiment@gfx1300', logicalTestId: 'experiment', target: 'gfx1300', suite: 'experimental' });
  manual.targets.push('gfx1300');
  const data = loadDashboardData({
    ...benchmarkData,
    pluginRuns: [...benchmarkData.pluginRuns, manual],
    testCatalog: [...benchmarkData.testCatalog, { id: 'experiment', suite: 'experimental', name: manual.tests.at(-1).name, problem: manual.tests.at(-1).problem }],
  });
  return { data, manual };
}

test.each([['feature/experiment', false], ['develop', true]])('isolates manual %s runs from all official views', (branch, sameSha) => {
  const { data, manual } = withManualRun(branch, sameSha);
  expect(data.runs).toEqual(benchmarkData.runs);
  expect(data.latestRun).toEqual(benchmarkData.latestRun);
  expect(data.latestCommitRun).toEqual(benchmarkData.latestCommitRun);
  expect(data.backfillRunIds).toEqual(benchmarkData.backfillRunIds);
  expect(data.testCatalog).toEqual(benchmarkData.testCatalog);
  expect(data.targets).toEqual(benchmarkData.targets);
  expect(data.suites).toEqual(benchmarkData.suites);
  expect(data.comparisonTargets).toContain('gfx1300');
  expect(data.comparisonSuites).toContain('experimental');
  expect(selectOverview(data, filters)).toEqual(selectOverview(benchmarkData, filters));
  expect(selectFailures(data, filters)).toEqual(selectFailures(benchmarkData, filters));
  expect(selectRecentRuns(data, filters)).toEqual(selectRecentRuns(benchmarkData, filters));
  expect(selectBenchmarkSeries(data, filters, 'triton-gemm-f16-1024')).toEqual(selectBenchmarkSeries(benchmarkData, filters, 'triton-gemm-f16-1024'));
  expect(selectPluginComparisonGroups(data)).toEqual(selectPluginComparisonGroups(benchmarkData));
  expect(selectComparisonRuns(data, null, null, filters)).toEqual({ candidate: manual, baseline: benchmarkData.latestRun });
  expect(selectComparisonRuns(data, null, manual.runId, filters).baseline).toBe(manual);
  expect(selectComparisonRuns(data, manual.runId, null, { targets: ['gfx1300'], suites: ['experimental'] }).baseline).toBeNull();
});

test('manual plugin comparisons remain hidden from Plugin Comparison', () => {
  const data = loadDashboardData({ ...benchmarkData, pluginRuns: benchmarkData.pluginRuns.map((run) => ({ ...run, trigger: 'manual' })) });
  expect(selectPluginComparisonGroups(data)).toEqual([]);
  expect(data.runs).toEqual([]);
  expect(data.comparisonRuns).toHaveLength(79);
});

test('accepts a manual-only published branch dataset and keeps official state empty', () => {
  const run = structuredClone(publishedRuns.find((candidate) => candidate.plugin.id === 'vanilla'));
  run.source.branch = 'feature/experiment';
  run.execution.trigger = 'manual';
  const { data } = validatePublishedDashboardData({
    metadata: dataMetadata, index: { generatedAt: run.execution.completedAt, runFiles: [`runs/${run.id}.json`] },
    runs: [run], catalogs: publishedCatalogs,
  });
  expect(data.runs).toEqual([]);
  expect(data.latestRun).toBeNull();
  expect(data.latestCommitRun).toBeNull();
  expect(data.testCatalog).toEqual([]);
  expect(data.targets).toEqual([]);
  expect(data.suites).toEqual([]);
  expect(data.comparisonTargets).toContain('gfx1250');
  expect(data.comparisonSuites.length).toBeGreaterThan(0);
  expect(selectComparisonRuns(data, null, null, filters)).toMatchObject({ candidate: { branch: 'feature/experiment' }, baseline: null });
});

test('automatic candidates retain the existing default baseline rules', () => {
  const data = loadDashboardData({ ...benchmarkData, pluginRuns: benchmarkData.pluginRuns.filter((run) => run.trigger === 'auto') });
  expect(selectComparisonRuns(data, null, null, filters)).toEqual({
    candidate: data.latestRun, baseline: data.runs.at(-2),
  });
  const selected = selectComparisonRuns(data, data.latestRun.runId, null, filters);
  expect(selected.candidate).toBe(benchmarkData.latestRun);
  expect(selected.baseline.trigger).toBe('auto');
  expect(Date.parse(selected.baseline.commitTimestamp)).toBeLessThan(Date.parse(selected.candidate.commitTimestamp));
});
