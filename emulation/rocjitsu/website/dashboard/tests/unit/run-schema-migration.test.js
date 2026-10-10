import { loadDashboardDataFiles } from '../../src/data/dashboardData.js';
import { expect, test } from 'vitest';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';

function publication() {
  const run = {
    schemaVersion: 2, id: 'new-run', testCatalog: 'test-catalogs/current.json',
    source: { branch: 'develop', commit: 'a'.repeat(40), committedAt: '2026-01-01T00:00:00Z' },
    execution: { completedAt: '2026-01-01T01:00:00Z', trigger: 'auto', machine: 'fictional' },
    environment: [],
    configurations: [{ target: 'gfx950', threadingMode: 'MT', results: [{ testId: 'a', status: 'completed', durationSeconds: 2, error: null }] }],
  };
  return {
    index: { generatedAt: '2026-01-02T00:00:00Z', runFiles: ['runs/default-branch/new-run.json'] },
    runs: [run],
    catalogs: { 'test-catalogs/current.json': { id: 'current', tests: [{ id: 'a', suite: 'Fictional', name: 'Example', problem: {} }], configurations: { 'gfx950:MT': ['a'] } } },
  };
}

test('new runs declare their own schema and use branch-class directories without index/catalog versions', () => {
  const source = publication();
  expect(validatePublishedDashboardData(source).data.runs[0].tests[0].mode).toBe('MT');
  source.runs[0].source.branch = 'feature/example';
  source.index.runFiles = ['runs/side-branches/new-run.json'];
  const { data, sourceData } = validatePublishedDashboardData(source);
  expect(data.runs).toEqual([]);
  expect(data.allRuns).toHaveLength(1);
  expect(sourceData).toEqual(source);
});

test('schema-2 runs sharing a catalog preserve its published configuration order', () => {
  const source = publication();
  const run = source.runs[0];
  const catalogFile = run.testCatalog;
  const catalog = source.catalogs[catalogFile];
  run.configurations = [
    { target: 'gfx950', threadingMode: 'ST', results: structuredClone(run.configurations[0].results) },
    ...run.configurations,
  ];
  catalog.configurations = {
    'gfx950:ST': ['a'],
    'gfx950:MT': ['a'],
  };
  const second = structuredClone(run);
  second.id = 'second-run';
  source.runs.push(second);
  source.index.runFiles.push('runs/default-branch/second-run.json');
  const { data } = validatePublishedDashboardData(source);
  expect(Object.keys(data.catalogs[catalogFile].configurations)).toEqual([
    'gfx950:ST',
    'gfx950:MT',
  ]);
});

function legacyPublication() {
  const source = publication();
  const run = source.runs[0];
  run.schemaVersion = 1;
  run.plugin = { id: 'vanilla', name: 'Vanilla' };
  run.comparisonId = 'legacy-comparison';
  run.targets = run.configurations.map(({ target, results }) => ({ id: target, results }));
  delete run.configurations;
  const catalog = source.catalogs[run.testCatalog];
  catalog.targets = { gfx950: ['a'] };
  delete catalog.configurations;
  source.index.runFiles = ['runs/new-run.json'];
  return source;
}

test.each([true, false])('schema-1 Vanilla falls back to MT without thread-count provenance and preserves raw input (declared=%s)', (declared) => {
  const source = legacyPublication();
  if (!declared) delete source.runs[0].schemaVersion;
  const before = structuredClone(source);
  const { data, sourceData } = validatePublishedDashboardData(source);
  expect(data.runs[0].runId).toBe(source.runs[0].id);
  expect(data.runs[0].tests[0]).toMatchObject({ testId: 'gfx950:MT:a', logicalTestId: 'a', target: 'gfx950', mode: 'MT', durationSeconds: 2 });
  expect(data.catalogs['test-catalogs/current.json'].configurations).toEqual({ 'gfx950:MT': ['a'] });
  expect(sourceData).toEqual(before);
  expect(source).toEqual(before);
});

test('schema-1 migration uses recorded per-target thread counts and preserves raw input', () => {
  const source = legacyPublication();
  const run = source.runs[0];
  const catalog = source.catalogs[run.testCatalog];
  run.environment = [
    { key: 'target.gfx950.numThreads', label: 'target.gfx950.numThreads', value: 1 },
    { key: 'target.gfx1250.numThreads', label: 'target.gfx1250.numThreads', value: 8 },
  ];
  run.targets.push({ id: 'gfx1250', results: structuredClone(run.targets[0].results) });
  catalog.targets.gfx1250 = ['a'];
  const before = structuredClone(source);
  const { data, sourceData } = validatePublishedDashboardData(source);
  expect(data.runs[0].tests).toEqual(expect.arrayContaining([
    expect.objectContaining({ testId: 'gfx950:ST:a', target: 'gfx950', mode: 'ST' }),
    expect.objectContaining({ testId: 'gfx1250:MT:a', target: 'gfx1250', mode: 'MT' }),
  ]));
  expect(data.catalogs[run.testCatalog].configurations).toEqual({
    'gfx950:ST': ['a'],
    'gfx1250:MT': ['a'],
  });
  expect(sourceData).toEqual(before);
  expect(source).toEqual(before);
});

test.each([
  [1, 8],
  [8, 1],
])('schema-1 runs sharing a catalog preserve ST and MT memberships regardless of order (%s then %s)', (firstThreads, secondThreads) => {
  const source = legacyPublication();
  const catalogFile = source.runs[0].testCatalog;
  const makeRun = (id, threads) => {
    const run = structuredClone(source.runs[0]);
    run.id = id;
    run.environment = [
      { key: 'target.gfx950.numThreads', label: 'target.gfx950.numThreads', value: threads },
    ];
    return run;
  };
  source.runs = [makeRun('first-run', firstThreads), makeRun('second-run', secondThreads)];
  source.index.runFiles = ['runs/first-run.json', 'runs/second-run.json'];
  const { data } = validatePublishedDashboardData(source);
  expect(data.catalogs[catalogFile].configurations).toEqual({
    'gfx950:ST': ['a'],
    'gfx950:MT': ['a'],
  });
  expect(data.allRuns.map((run) => run.modes)).toEqual(expect.arrayContaining([['ST'], ['MT']]));
});

test.each([
  ['duplicate', [1, 1]],
  ['conflicting', [1, 8]],
])('schema-1 migration rejects %s thread-count provenance', (_label, values) => {
  const source = legacyPublication();
  source.runs[0].environment = values.map((value) => ({
    key: 'target.gfx950.numThreads', label: 'target.gfx950.numThreads', value,
  }));
  expect(() => validatePublishedDashboardData(source)).toThrow(/numThreads.*exactly once/i);
});

test.each([0, -1, 1.5, '1', null])('schema-1 migration rejects invalid thread count %s', (value) => {
  const source = legacyPublication();
  source.runs[0].environment = [
    { key: 'target.gfx950.numThreads', label: 'target.gfx950.numThreads', value },
  ];
  expect(() => validatePublishedDashboardData(source)).toThrow(/numThreads.*positive integer/i);
});

test('non-Vanilla files are skipped rather than relabeled as ordinary measurements', () => {
  const source = legacyPublication();
  source.runs[0].plugin.id = 'asan';
  const { data, sourceData } = validatePublishedDashboardData(source);
  expect(data.allRuns).toEqual([]);
  expect(data.testCatalog).toEqual([]);
  expect(sourceData.runs).toEqual(source.runs);
});

test('loader fetches no metadata or excluded-plugin catalog and retains skipped raw files', async () => {
  const source = legacyPublication();
  const plugin = { ...structuredClone(source.runs[0]), id: 'asan-run', plugin: { id: 'asan', name: 'ASan' }, testCatalog: 'test-catalogs/plugins.json' };
  source.runs.push(plugin);
  source.index.runFiles.push('runs/asan-run.json');
  const bodies = { 'index.json': source.index, ...source.catalogs, ...Object.fromEntries(source.runs.map((run, i) => [source.index.runFiles[i], run])) };
  const requests = [];
  const result = await loadDashboardDataFiles({
    indexUrl: 'https://example.test/data/index.json',
    fetch: async (url) => {
      const path = new URL(url).pathname.replace('/data/', '');
      requests.push(path);
      return { ok: Object.hasOwn(bodies, path), status: Object.hasOwn(bodies, path) ? 200 : 404, text: async () => JSON.stringify(bodies[path]) };
    },
  });
  expect(requests).not.toContain('metadata.json');
  expect(requests).not.toContain('test-catalogs/plugins.json');
  expect(result.data.allRuns).toHaveLength(1);
  expect(result.data.allRuns[0].tests[0].mode).toBe('MT');
  expect(result.sourceData.runs).toEqual(source.runs);
});

test.each([undefined, null, 0, 3, 99, '2'])('unsupported current run version %s fails the whole load', (version) => {
  const source = publication();
  if (version === undefined) delete source.runs[0].schemaVersion;
  else source.runs[0].schemaVersion = version;
  expect(() => validatePublishedDashboardData(source)).toThrow(version === undefined ? /schema-1.*contract/i : /unsupported schema version/i);
});

test.each([
  'runs/new-run.json',
  'runs/side-branches/new-run.json',
  'runs/side-branches/feature/new-run.json',
  'runs/default-branch/../new-run.json',
])('current run rejects wrong, nested, or unsafe directory: %s', (path) => {
  const source = publication(); source.index.runFiles = [path];
  expect(() => validatePublishedDashboardData(source)).toThrow(/filename|branch directory/i);
});

test.each([
  ['missing plugin', (run) => { delete run.plugin; }],
  ['invalid plugin', (run) => { run.plugin = {}; }],
  ['mixed legacy and current groups', (run) => { run.configurations = []; }],
  ['duplicate target', (run) => { run.targets.push(structuredClone(run.targets[0])); }],
  ['missing test result', (run) => { run.targets[0].results = []; }],
  ['unknown status', (run) => { run.targets[0].results[0].status = 'unknown'; }],
  ['ambiguous error', (run) => { run.targets[0].results[0].error = 'not clean'; }],
  ['nonpositive legacy duration', (run) => { run.targets[0].results[0].durationSeconds = 0; }],
])('invalid schema-1 Vanilla data fails rather than disappearing: %s', (_label, mutate) => {
  const source = legacyPublication(); mutate(source.runs[0]);
  expect(() => validatePublishedDashboardData(source)).toThrow();
});

test('native and legacy Vanilla measurements coexist; legacy side branch is MT', () => {
  const current = publication();
  const legacy = legacyPublication();
  const old = legacy.runs[0];
  old.id = 'old-run'; old.source.branch = 'feature/old'; old.testCatalog = 'test-catalogs/legacy.json';
  const catalog = { ...legacy.catalogs['test-catalogs/current.json'], id: 'legacy' };
  current.runs.push(old);
  current.index.runFiles.push('runs/side-branches/old-run.json');
  current.catalogs[old.testCatalog] = catalog;
  const { data } = validatePublishedDashboardData(current);
  expect(data.allRuns.find((run) => run.runId === 'old-run').tests[0].mode).toBe('MT');
  expect(data.allRuns.find((run) => run.runId === 'new-run').tests[0].mode).toBe('MT');
  expect(data.runs).toHaveLength(1);
  expect(data.allRuns).toHaveLength(2);
});

test('schema-2 cannot hide legacy target measurements alongside current groups', () => {
  const source = publication();
  source.runs[0].targets = [{ id: 'gfx950', results: [] }];
  expect(() => validatePublishedDashboardData(source)).toThrow(/schema|target/i);
});
