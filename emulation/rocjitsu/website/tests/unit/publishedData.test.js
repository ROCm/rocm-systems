import { describe, expect, test } from 'vitest';
import {
  loadDashboardData,
  validatePublishedDashboardData,
  validatePublishedResult,
} from '../../src/data/dashboardValidation.js';
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
  expect(publishedResult.publicationIssues).toEqual([
    {
      runId: 'benchmark-202608270530-0db03af1',
      message: 'Run benchmark-202608270530-0db03af1 uses machine sjc-rocjitsu-perf-02; expected sjc-rocjitsu-perf-01',
    },
    {
      runId: 'benchmark-202608290530-19872076',
      message: 'Run benchmark-202608290530-19872076 uses a different environment than benchmark-202606010530-86b362ea',
    },
  ]);
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
  test.each([
    ['a missing testId', {
      durationSeconds: 1.25,
      status: 'completed',
      error: null,
    }, 'Result must contain a testId'],
    ['a removed legacy field', {
      testId: 'generated-test',
      durationSeconds: null,
      status: 'failed',
      error: 'Process failed',
      exitCode: 1,
    }, 'Result generated-test contains removed field exitCode'],
    ['a removed findings field', {
      testId: 'generated-test',
      durationSeconds: null,
      status: 'failed',
      error: 'Process failed',
      findings: [],
    }, 'Result generated-test contains removed field findings'],
    ['an invalid status', {
      testId: 'generated-test',
      durationSeconds: null,
      status: 'skipped',
      error: null,
    }, 'Result generated-test has invalid status skipped'],
    ['a missing durationSeconds', {
      testId: 'generated-test',
      status: 'failed',
      error: 'Process failed',
    }, 'Result generated-test must contain durationSeconds'],
    ['a non-positive completed duration', {
      testId: 'generated-test',
      durationSeconds: 0,
      status: 'completed',
      error: null,
    }, 'Completed result generated-test must contain a positive durationSeconds'],
    ['a duration on a failed result', {
      testId: 'generated-test',
      durationSeconds: 1.25,
      status: 'failed',
      error: 'Process failed',
    }, 'failed result generated-test must have a null durationSeconds'],
    ['a missing error', {
      testId: 'generated-test',
      durationSeconds: null,
      status: 'timeout',
    }, 'Result generated-test must contain error'],
    ['an error on a completed result', {
      testId: 'generated-test',
      durationSeconds: 1.25,
      status: 'completed',
      error: 'Unexpected diagnostic',
    }, 'Completed result generated-test cannot contain an error'],
    ['an empty error', {
      testId: 'generated-test',
      durationSeconds: null,
      status: 'failed',
      error: '',
    }, 'Result generated-test error must be a non-empty string or null'],
  ])('rejects a standalone result with %s', (_, result, expectedError) => {
    expect(() => validatePublishedResult(result)).toThrow(expectedError);
  });

  test('accepts a failed standalone result with a null duration', () => {
    expect(validatePublishedResult({
      testId: 'generated-test',
      durationSeconds: null,
      status: 'failed',
      error: 'Process failed',
    })).toEqual({
      testId: 'generated-test',
      durationSeconds: null,
      status: 'failed',
      error: 'Process failed',
    });
  });

  test('keeps a normalized run outside the publication policy', () => {
    const data = cloneBenchmarkData();
    data.runs[0].branch = 'feature/experiment';

    expect(loadDashboardData(data).runs[0].branch).toBe('feature/experiment');
  });

  test('reports publication policy differences without skipping runs', () => {
    const runFiles = dataIndex.runFiles.slice(0, 2);
    const runs = structuredClone(publishedRuns.slice(0, 2));
    runs[1].source.branch = 'feature/experiment';
    runs[1].execution.machine = 'different-runner';
    runs[1].environment[0].value = 'different-environment';

    const result = validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs: publishedCatalogs,
    });

    expect(result.data.runs).toHaveLength(2);
    expect(result.warnings).toEqual([]);
    expect(result.publicationIssues).toEqual([
      {
        runId: runs[1].id,
        message: `Run ${runs[1].id} must use source branch develop`,
      },
      {
        runId: runs[1].id,
        message: `Run ${runs[1].id} uses machine different-runner; expected ${runs[0].execution.machine}`,
      },
      {
        runId: runs[1].id,
        message: `Run ${runs[1].id} uses a different environment than ${runs[0].id}`,
      },
    ]);
  });

  test('requires metadata to contain a repository URL', () => {
    expect(() => validatePublishedDashboardData({
      metadata: { ...dataMetadata, repository: '' },
      index: dataIndex,
      runs: publishedRuns,
      runErrors: publishedRunErrors,
      catalogs: publishedCatalogs,
    })).toThrow('Expected dashboard metadata to contain an HTTP repository URL');
  });

  test('requires an ISO-8601 index timestamp', () => {
    expect(() => validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { ...dataIndex, generatedAt: '2026' },
      runs: publishedRuns,
      runErrors: publishedRunErrors,
      catalogs: publishedCatalogs,
    })).toThrow('Expected the dashboard data index to contain generatedAt and runFiles');
  });

  test('requires a run filename to match its run ID', () => {
    const runFiles = dataIndex.runFiles.slice(0, 2);
    const runs = structuredClone(publishedRuns.slice(0, 2));
    runFiles[1] = 'runs/a-different-run-id.json';

    const result = validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs: publishedCatalogs,
    });

    expect(result.data.runs).toHaveLength(1);
    expect(result.warnings).toEqual([{
      runFile: runFiles[1],
      message: `Run ${runs[1].id} must be published as runs/${runs[1].id}.json`,
    }]);
  });

  test('requires a catalog filename to match its catalog ID', () => {
    const runFiles = dataIndex.runFiles.slice(0, 2);
    const runs = structuredClone(publishedRuns.slice(0, 2));
    const catalogs = structuredClone(publishedCatalogs);
    const mismatchedPath = 'test-catalogs/a-different-catalog-id.json';
    catalogs[mismatchedPath] = catalogs[runs[1].testCatalog];
    runs[1].testCatalog = mismatchedPath;

    const result = validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs,
    });

    expect(result.data.runs).toHaveLength(1);
    expect(result.warnings).toEqual([{
      runFile: runFiles[1],
      message: `Test catalog ${mismatchedPath} must contain id a-different-catalog-id`,
    }]);
  });

  test.each([
    ['catalog ID', (catalog) => { catalog.id = ''; }, 'does not match the schema-version-1 contract'],
    ['test ID', (catalog) => { catalog.tests[0].id = ''; }, 'contains an invalid or duplicate test definition'],
    ['test suite', (catalog) => { catalog.tests[0].suite = ''; }, 'contains an invalid or duplicate test definition'],
    ['test name', (catalog) => { catalog.tests[0].name = ''; }, 'contains an invalid or duplicate test definition'],
    ['problem object', (catalog) => { catalog.tests[0].problem = null; }, 'contains an invalid or duplicate test definition'],
    ['target test IDs', (catalog) => {
      catalog.targets[Object.keys(catalog.targets)[0]][0] = 'unknown-test';
    }, 'contains an invalid test set'],
  ])('rejects an invalid %s in a referenced catalog', (_, makeInvalid, expectedMessage) => {
    const runFiles = dataIndex.runFiles.slice(0, 2);
    const runs = structuredClone(publishedRuns.slice(0, 2));
    const catalogs = structuredClone(publishedCatalogs);
    const catalogPath = 'test-catalogs/validation-audit.json';
    catalogs[catalogPath] = structuredClone(catalogs[runs[1].testCatalog]);
    catalogs[catalogPath].id = 'validation-audit';
    runs[1].testCatalog = catalogPath;
    makeInvalid(catalogs[catalogPath]);

    const result = validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs,
    });

    expect(result.data.runs).toHaveLength(1);
    expect(result.warnings).toHaveLength(1);
    expect(result.warnings[0]).toMatchObject({ runFile: runFiles[1] });
    expect(result.warnings[0].message).toContain(expectedMessage);
  });

  test.each([
    ['id', (run) => { run.id = ''; }],
    ['comparison ID', (run) => { run.comparisonId = ''; }],
    ['plugin ID', (run) => { run.plugin.id = ''; }],
    ['plugin name', (run) => { run.plugin.name = ''; }],
    ['plugin version', (run) => { run.plugin.version = ''; }],
    ['plugin options', (run) => { run.plugin.options = []; }],
    ['branch', (run) => { run.source.branch = ''; }],
    ['full commit SHA', (run) => { run.source.commit = '1234abcd'; }],
    ['commit timestamp', (run) => { run.source.committedAt = 'not-a-date'; }],
    ['optional commit message when present', (run) => { run.source.message = ''; }],
    ['completion timestamp', (run) => { run.execution.completedAt = 'not-a-date'; }],
    ['trigger', (run) => { run.execution.trigger = 'scheduled'; }],
    ['machine', (run) => { run.execution.machine = ''; }],
    ['environment key', (run) => { run.environment[0].key = ''; }],
    ['environment label', (run) => { run.environment[0].label = ''; }],
    ['environment value', (run) => { run.environment[0].value = ''; }],
    ['unique environment key', (run) => { run.environment[1].key = run.environment[0].key; }],
  ])('rejects an invalid run %s', (_, makeInvalid) => {
    const runFiles = dataIndex.runFiles.slice(0, 2);
    const runs = structuredClone(publishedRuns.slice(0, 2));
    makeInvalid(runs[1]);

    const result = validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs: publishedCatalogs,
    });

    expect(result.data.runs).toHaveLength(1);
    expect(result.warnings).toEqual([{
      runFile: runFiles[1],
      message: `Run ${runs[1].id || '(unknown)'} does not match the schema-version-1 run contract`,
    }]);
  });

  test.each([
    ['target ID', (run) => { run.targets[0].id = ''; }, 'does not contain exactly the targets required'],
    ['unique target ID', (run) => { run.targets[1].id = run.targets[0].id; }, 'does not contain exactly the targets required'],
    ['results array', (run) => { run.targets[0].results = null; }, 'does not contain exactly the targets required'],
    ['complete results', (run) => { run.targets[0].results.pop(); }, 'does not contain exactly one valid result'],
    ['unique result testId', (run) => {
      run.targets[0].results[1].testId = run.targets[0].results[0].testId;
    }, 'does not contain exactly one valid result'],
    ['catalog result testId', (run) => {
      run.targets[0].results[0].testId = 'unknown-test';
    }, 'does not contain exactly one valid result'],
    ['result with required fields', (run) => {
      delete run.targets[0].results[0].error;
    }, 'must contain error'],
    ['positive completed result duration', (run) => {
      const completedResult = run.targets[0].results.find((result) => result.status === 'completed');
      completedResult.durationSeconds = -1;
    }, 'must contain a positive durationSeconds'],
  ])('requires every run to have a valid %s', (_, makeInvalid, expectedMessage) => {
    const runFiles = dataIndex.runFiles.slice(0, 2);
    const runs = structuredClone(publishedRuns.slice(0, 2));
    makeInvalid(runs[1]);

    const result = validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs: publishedCatalogs,
    });

    expect(result.data.runs).toHaveLength(1);
    expect(result.warnings).toHaveLength(1);
    expect(result.warnings[0]).toMatchObject({ runFile: runFiles[1] });
    expect(result.warnings[0].message).toContain(expectedMessage);
  });

  test('accepts an empty environment array', () => {
    const runs = structuredClone(publishedRuns);
    runs[0].environment = [];
    expect(() => validatePublishedDashboardData({
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

    const result = validatePublishedDashboardData({
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

    expect(() => validatePublishedDashboardData({
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

  const result = validatePublishedDashboardData({
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
