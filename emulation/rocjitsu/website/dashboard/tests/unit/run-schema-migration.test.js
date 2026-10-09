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

test.each([true, false])('schema-1 Vanilla migrates target groups to MT and preserves raw input (declared=%s)', (declared) => {
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
