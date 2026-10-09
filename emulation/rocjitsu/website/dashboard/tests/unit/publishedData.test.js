import { describe, expect, test } from 'vitest';
import {
  loadDashboardData,
  validatePublishedDashboardData,
  validatePublishedResult,
} from '../../src/data/dashboardValidation.js';
import { selectPluginComparisonGroups } from '../../src/data/pluginComparison.js';
import { isRunCompletedForFilters } from '../../src/data/selectors.js';
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
  expect(publishedResult.sourceData.runs).toHaveLength(83);
  expect(dataMetadata.schemaVersion).toBe(1);
  expect(dataIndex.runFiles).toHaveLength(83);
  expect(dataIndex.runFiles.every((runFile) => /^runs\/[^/]+\.json$/.test(runFile))).toBe(true);
  expect(Object.keys(publishedCatalogs).sort()).toEqual([
    'test-catalogs/rocjitsu-core-v1.json',
    'test-catalogs/rocjitsu-core-v2.json',
  ]);
  expect(benchmarkData.runs).toHaveLength(79);
  expect(benchmarkData.pluginRuns).toHaveLength(83);
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
  const currentTargetTests = benchmarkData.latestCommitRun.tests.filter((test) => test.target === 'gfx1250');
  expect(historicalTargetTests).toHaveLength(5);
  expect(currentTargetTests).toHaveLength(7);
  expect(benchmarkData.testCatalog).toHaveLength(7);
  expect(isRunCompletedForFilters(historicalRun, {
    targets: ['gfx1250'],
    suites: benchmarkData.suites,
  })).toBe(true);
});

test('keeps v1 smaller and faster than the v2 workload', () => {
  const workloadByCatalog = new Map();
  for (const run of benchmarkData.runs) {
    const workload = workloadByCatalog.get(run.catalogId) ?? {
      testIds: new Set(),
      totals: [],
    };
    run.tests.forEach((test) => workload.testIds.add(test.logicalTestId));
    for (const target of new Set(run.tests.map((test) => test.target))) {
      workload.totals.push(run.tests
        .filter((test) => test.target === target)
        .reduce((total, test) => total + (test.durationSeconds ?? 0), 0));
    }
    workloadByCatalog.set(run.catalogId, workload);
  }

  const v1 = workloadByCatalog.get('rocjitsu-core-v1');
  const v2 = workloadByCatalog.get('rocjitsu-core-v2');
  expect(v1.testIds.size).toBeLessThan(v2.testIds.size);
  expect(Math.max(...v1.totals)).toBeLessThan(Math.min(...v2.totals));
  expect(Math.max(...v1.totals)).toBeGreaterThan(850);
  expect(Math.max(...v1.totals)).toBeLessThan(950);
  expect(Math.max(...v2.totals)).toBeGreaterThan(1_100);
  expect(Math.max(...v2.totals)).toBeLessThan(1_300);
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

  test('allows environments to change between independent historical runs', () => {
    const runFiles = dataIndex.runFiles.slice(0, 2);
    const runs = structuredClone(publishedRuns.slice(0, 2));
    runs[1].environment[0].value = 'different-environment';

    const result = validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs: publishedCatalogs,
    });

    expect(result.data.runs).toHaveLength(2);
  });

  test('rejects branch and machine publication policy violations', () => {
    const runFiles = dataIndex.runFiles.slice(0, 2);
    const runs = structuredClone(publishedRuns.slice(0, 2));
    runs[1].source.branch = 'feature/experiment';
    runs[1].execution.machine = 'different-runner';

    expect(() => validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs: publishedCatalogs,
    })).toThrow(/must use source branch develop[\s\S]*uses machine different-runner/);
  });

  test.each([
    ['branch', (run) => { run.source.branch = 'feature/plugin-test'; }, 'does not match comparison'],
    ['commit', (run) => { run.source.commit = 'f'.repeat(40); }, 'does not match comparison'],
    [
      'commit timestamp',
      (run) => { run.source.committedAt = '2026-07-25T00:00:00.000Z'; },
      'has conflicting committedAt values',
    ],
    [
      'trigger',
      (run) => { run.execution.trigger = run.execution.trigger === 'auto' ? 'manual' : 'auto'; },
      'does not match comparison',
    ],
    ['machine', (run) => { run.execution.machine = 'different-runner'; }, 'does not match comparison'],
    ['environment', (run) => { run.environment[0].value = 'different-environment'; }, 'does not match comparison'],
  ])('rejects a plugin comparison with a different %s', (_, mutatePlugin, expectedMessage) => {
    const comparisonId = 'benchmark-202607250530-8e0c5183';
    const runs = structuredClone(publishedRuns.filter((run) => run.comparisonId === comparisonId));
    const pluginRun = runs.find((run) => run.plugin.id !== 'vanilla');
    mutatePlugin(pluginRun);

    expect(() => validatePublishedDashboardData({
      metadata: dataMetadata,
      index: {
        generatedAt: dataIndex.generatedAt,
        runFiles: runs.map((run) => `runs/${run.id}.json`),
      },
      runs,
      catalogs: publishedCatalogs,
    })).toThrow(expectedMessage);
  });

  test('rejects an instrumented plugin comparison without a Vanilla baseline', () => {
    const comparisonId = 'benchmark-202607250530-8e0c5183';
    const runs = structuredClone(publishedRuns.filter((run) => (
      run.comparisonId === comparisonId && run.plugin.id !== 'vanilla'
    )));

    expect(() => validatePublishedDashboardData({
      metadata: dataMetadata,
      index: {
        generatedAt: dataIndex.generatedAt,
        runFiles: runs.map((run) => `runs/${run.id}.json`),
      },
      runs,
      catalogs: publishedCatalogs,
    })).toThrow(`Comparison ${comparisonId} has plugin runs without a Vanilla baseline`);
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

    expect(() => validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs: publishedCatalogs,
    })).toThrow(`Run ${runs[1].id} must be published as runs/${runs[1].id}.json`);
  });

  test('requires a catalog filename to match its catalog ID', () => {
    const runFiles = dataIndex.runFiles.slice(0, 2);
    const runs = structuredClone(publishedRuns.slice(0, 2));
    const catalogs = structuredClone(publishedCatalogs);
    const mismatchedPath = 'test-catalogs/a-different-catalog-id.json';
    catalogs[mismatchedPath] = catalogs[runs[1].testCatalog];
    runs[1].testCatalog = mismatchedPath;

    expect(() => validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs,
    })).toThrow(`Test catalog ${mismatchedPath} must contain id a-different-catalog-id`);
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
    ['orphan test definition', (catalog) => {
      catalog.tests.push({
        id: 'unreferenced-test',
        suite: 'validation',
        name: 'Unreferenced test',
        problem: {},
      });
    }, 'defines unreferenced-test without assigning it to a target'],
  ])('rejects an invalid %s in a referenced catalog', (_, makeInvalid, expectedMessage) => {
    const runFiles = dataIndex.runFiles.slice(0, 2);
    const runs = structuredClone(publishedRuns.slice(0, 2));
    const catalogs = structuredClone(publishedCatalogs);
    const catalogPath = 'test-catalogs/validation-audit.json';
    catalogs[catalogPath] = structuredClone(catalogs[runs[1].testCatalog]);
    catalogs[catalogPath].id = 'validation-audit';
    runs[1].testCatalog = catalogPath;
    makeInvalid(catalogs[catalogPath]);

    expect(() => validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs,
    })).toThrow(expectedMessage);
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

    expect(() => validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs: publishedCatalogs,
    })).toThrow(`Run ${runs[1].id || '(unknown)'} does not match the schema-version-1 run contract`);
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

    expect(() => validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs: publishedCatalogs,
    })).toThrow(expectedMessage);
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

  test('rejects a run when one commit has conflicting committedAt values', () => {
    const runFiles = dataIndex.runFiles.slice(0, 2);
    const runs = structuredClone(publishedRuns.slice(0, 2));
    runs[1].source.commit = runs[0].source.commit;

    expect(() => validatePublishedDashboardData({
      metadata: dataMetadata,
      index: { generatedAt: dataIndex.generatedAt, runFiles },
      runs,
      catalogs: publishedCatalogs,
    })).toThrow(`Commit ${runs[0].source.commit} has conflicting committedAt values`);
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

test('rejects an invalid published run with its filename and reason', () => {
  const runFiles = dataIndex.runFiles.slice(0, 2);
  const runs = structuredClone(publishedRuns.slice(0, 2));
  runs[1].targets[1].id = 'gfx1250';

  expect(() => validatePublishedDashboardData({
    metadata: dataMetadata,
    index: { generatedAt: dataIndex.generatedAt, runFiles },
    runs,
    catalogs: publishedCatalogs,
  })).toThrow(
    `Run ${runs[1].id} does not contain exactly the targets required by rocjitsu-core-v1`,
  );
});
