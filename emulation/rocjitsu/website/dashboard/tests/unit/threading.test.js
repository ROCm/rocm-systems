import { expect, test } from 'vitest';
import { benchmarkData } from '../fixtures/publishedData.js';
import { loadDashboardData, selectThreadingModeData } from '../../src/data/dashboardValidation.js';
import { compareRuns, previousCompletedRunForFilters, selectOverview, selectFailures } from '../../src/data/selectors.js';
import { selectPluginComparison, selectPluginComparisonGroups } from '../../src/data/pluginComparison.js';
import { reconcileSelection } from '../../src/hooks/useDashboardState.js';

const filters = { targets: ['gfx1250'], suites: benchmarkData.suites };

test('filter reconciliation preserves intentional empty selections and falls back only when all choices disappear', () => {
  expect(reconcileSelection([], ['new'], ['new'])).toEqual([]);
  expect(reconcileSelection(['old', 'keep'], ['keep', 'new'], ['new'])).toEqual(['keep']);
  expect(reconcileSelection(['old'], ['new'], ['new'])).toEqual(['new']);
});

test('cross-mode inputs cannot compare or supply an earlier baseline', () => {
  const candidate = { ...benchmarkData.runs.at(-1), threadingMode: 'single' };
  const baseline = { ...benchmarkData.runs[0], threadingMode: 'default' };
  expect(compareRuns(candidate, baseline, filters).every((row) => !row.comparable)).toBe(true);
  expect(previousCompletedRunForFilters([baseline], candidate, filters)).toBeNull();
});

test('default historical comparisons permit changed allocation and configuration hashes', () => {
  const original = benchmarkData.runs[0];
  const candidate = { ...original, threadingMode: 'default', environmentId: 'new-allocation-hash',
    provenance: { ...original.provenance, details: [{ key: 'threadAllocation', label: 'Thread allocation', value: '2/2/4' }] } };
  expect(compareRuns(candidate, original, filters).every((row) => row.comparable && row.delta === 0)).toBe(true);
});

test('mode histories retain their own retries, failures, baselines and estimates on shared commits', () => {
  const singleRuns = benchmarkData.pluginRuns.filter((run) => run.plugin.id === 'vanilla').map((run) => ({
    ...run, runId: `${run.runId}-single`, comparisonId: `${run.comparisonId}-single`, threadingMode: 'single',
    tests: run.tests.map((result) => ({ ...result, durationSeconds: result.durationSeconds == null ? null : result.durationSeconds * 3 })),
  }));
  const combined = loadDashboardData({ ...benchmarkData, pluginRuns: [...benchmarkData.pluginRuns, ...singleRuns] });
  const defaults = selectThreadingModeData(combined, 'default');
  const singles = selectThreadingModeData(combined, 'single');
  expect(selectOverview(defaults, filters)).toEqual(selectOverview(benchmarkData, filters));
  expect(singles.runs).toHaveLength(benchmarkData.runs.length);
  expect([...singles.backfillRunIds]).toEqual([...benchmarkData.backfillRunIds].map((id) => `${id}-single`));
  expect(selectFailures(singles, filters)).toHaveLength(selectFailures(defaults, filters).length);
  const overview = selectOverview(singles, filters);
  expect(overview.candidate.threadingMode).toBe('single');
  expect(overview.baseline.threadingMode).toBe('single');
  expect(overview.history.currentDuration).toBeCloseTo(selectOverview(defaults, filters).history.currentDuration * 3);
});

test('plugin groups and direct comparison reject cross-mode pairing', () => {
  const group = selectPluginComparisonGroups(benchmarkData)[0];
  const vanilla = group.runs.find((run) => run.plugin.id === 'vanilla');
  const instrumented = { ...group.runs.find((run) => run.plugin.id !== 'vanilla'), threadingMode: 'single' };
  expect(selectPluginComparisonGroups({ pluginRuns: [vanilla, instrumented] })).toEqual([]);
  const comparison = selectPluginComparison({ runs: [vanilla, instrumented] }, 'gfx1250', filters.suites);
  expect(comparison.summaries.find(({ run }) => run === instrumented).comparable).toBe(0);
});

test('multi-mode selection retains both datasets and permits an empty selection', () => {
  const single = { ...benchmarkData.runs.at(-1), runId: 'single', comparisonId: 'single', threadingMode: 'single' };
  const combined = loadDashboardData({ ...benchmarkData, pluginRuns: [...benchmarkData.pluginRuns, single] });
  expect(selectThreadingModeData(combined, ['default', 'single']).runs).toHaveLength(benchmarkData.runs.length + 1);
  const empty = selectThreadingModeData(combined, []);
  expect(empty.runs).toEqual([]);
  expect(empty.testCatalog).toEqual([]);
});
