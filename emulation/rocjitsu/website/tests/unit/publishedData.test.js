import { describe, expect, test } from 'vitest';
import { loadDashboardData, loadPublishedDashboardData } from '../../src/data/dashboardData.js';
import { selectPluginComparisonGroups } from '../../src/data/pluginComparison.js';
import { isRunCompleted } from '../../src/data/runOrdering.js';
import { createDashboardFixture } from '../fixtures/dashboardFixture.js';
import {
  benchmarkData,
  cloneBenchmarkData,
  dataIndex,
  dataMetadata,
  publishedCatalogs,
  publishedRunErrors,
  publishedResult,
  publishedRuns,
} from '../fixtures/publishedData.js';

test('loads merged target runs from immutable test catalogs', () => {
  expect(publishedResult.warnings).toEqual([]);
  expect(dataMetadata.schemaVersion).toBe(1);
  expect(dataIndex.runFiles).toHaveLength(55);
  expect(dataIndex.runFiles.every((runFile) => /^runs\/[^/]+\.json$/.test(runFile))).toBe(true);
  expect(Object.keys(publishedCatalogs).sort()).toEqual([
    'test-catalogs/rocjitsu-core-v1.json',
    'test-catalogs/rocjitsu-core-v2.json',
  ]);
  expect(benchmarkData.runs).toHaveLength(51);
  expect(benchmarkData.pluginRuns).toHaveLength(55);
  expect(benchmarkData.runs.every((run) => run.plugin.id === 'vanilla')).toBe(true);
  expect(benchmarkData.runs.every((run) => (
    run.targets.includes('gfx1250') && run.targets.includes('gfx950')
  ))).toBe(true);
  expect(benchmarkData.runs.every((run) => (
    !Object.hasOwn(run, 'canonical')
    && run.branch === 'develop'
    && ['auto', 'manual'].includes(run.trigger)
  ))).toBe(true);

  const pluginGroups = selectPluginComparisonGroups(benchmarkData);
  expect(pluginGroups.map((group) => group.comparisonId)).toEqual([
    'benchmark-202607250530-8e0c5183',
    'benchmark-202608311945-31369c4d',
  ]);
  expect(pluginGroups[0].runs.map((run) => run.plugin.id)).toEqual(['vanilla', 'asan']);
  expect(pluginGroups[1].runs.map((run) => run.plugin.id)).toEqual(['vanilla', 'asan', 'tsan', 'ubsan']);
});

test('fixture generation is deterministic and isolates mutations between callers', () => {
  const first = createDashboardFixture();
  const second = createDashboardFixture();
  expect(second).toEqual(first);
  const generated = first.runs.filter((run) => run.id.startsWith('generated-history-'));
  expect(generated).toHaveLength(35);
  expect(new Set(first.runs.map((run) => run.id)).size).toBe(first.runs.length);
  expect(new Set(generated.map((run) => run.source.commit)).size).toBe(35);

  generated[0].targets[0].results[0].durationSeconds = -1;
  first.catalogs['test-catalogs/rocjitsu-core-v2.json'].tests[0].name = 'Mutated';
  expect(createDashboardFixture()).toEqual(second);
});

test('separates run completion time from tested commit time', () => {
  const run = benchmarkData.runs.find((candidate) => candidate.timestamp === '2026-08-31T13:10:00.000Z');
  expect({ runTime: run.timestamp, commitTime: run.commitTimestamp }).toEqual({
    runTime: '2026-08-31T13:10:00.000Z',
    commitTime: '2026-08-31T12:42:00.000Z',
  });
});

test('keeps an older smaller test set complete after the catalog grows', () => {
  const historicalRun = benchmarkData.runs.find((run) => run.runId === 'benchmark-202606010530-86b362ea');
  const historicalTargetTests = historicalRun.tests.filter((test) => test.target === 'gfx1250');
  const currentTargetTests = benchmarkData.latestCompletedRun.tests.filter((test) => test.target === 'gfx1250');
  expect(historicalTargetTests).toHaveLength(5);
  expect(currentTargetTests).toHaveLength(7);
  expect(benchmarkData.testCatalog).toHaveLength(7);
  expect(isRunCompleted(historicalRun)).toBe(true);
});

describe('dataset-level validation', () => {
  test('rejects data outside the official develop-run policy', () => {
    const invalidData = cloneBenchmarkData();
    invalidData.runs[0].branch = 'feature/experiment';
    expect(() => loadDashboardData(invalidData)).toThrow(/not a valid official develop run/);
  });

  test('accepts an empty environment array', () => {
    const runs = structuredClone(publishedRuns);
    runs[0].environment = [];
    expect(() => loadPublishedDashboardData({
      metadata: dataMetadata,
      index: dataIndex,
      runs,
      runErrors: publishedRunErrors,
      catalogs: publishedCatalogs,
    })).not.toThrow();
  });

  test('skips a run when one commit has conflicting committedAt values', () => {
    const runFiles = dataIndex.runFiles.slice(0, 2);
    const runs = structuredClone(publishedRuns.slice(0, 2));
    runs[1].source.commit = runs[0].source.commit;

    const result = loadPublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs: publishedCatalogs,
    });

    expect(result.data.runs).toHaveLength(1);
    expect(result.warnings).toEqual([{
      runFile: runFiles[1],
      message: `Commit ${runs[0].source.commit} has conflicting committedAt values`,
    }]);
  });

  test('rejects catalogs that reuse a test ID for a different workload', () => {
    const conflictingCatalogs = structuredClone(publishedCatalogs);
    conflictingCatalogs['test-catalogs/rocjitsu-core-v2.json'].tests
      .find((test) => test.id === 'triton-gemm-f16-1024').problem.m = 2048;

    expect(() => loadPublishedDashboardData({
      metadata: dataMetadata,
      index: dataIndex,
      runs: publishedRuns,
      runErrors: publishedRunErrors,
      catalogs: conflictingCatalogs,
    })).toThrow('Test triton-gemm-f16-1024 is defined differently by test-catalogs/rocjitsu-core-v1.json and test-catalogs/rocjitsu-core-v2.json');
  });
});

test('skips an invalid published run and reports its filename and reason', () => {
  const runFiles = dataIndex.runFiles.slice(0, 2);
  const runs = structuredClone(publishedRuns.slice(0, 2));
  runs[1].targets[1].id = 'gfx1250';

  const result = loadPublishedDashboardData({
    metadata: dataMetadata,
    index: { generatedAt: dataIndex.generatedAt, runFiles },
    runs,
    catalogs: publishedCatalogs,
  });

  expect(result.data.runs).toHaveLength(1);
  expect(result.data.runs[0].targets).toEqual(['gfx1250', 'gfx950']);
  expect(result.warnings).toEqual([{
    runFile: runFiles[1],
    message: `Run ${runs[1].id} does not contain exactly the targets required by rocjitsu-core-v1`,
  }]);
});
