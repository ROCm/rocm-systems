import { expect, test } from 'vitest';
import { selectAutomaticReference, selectPublishedBranches } from '../../src/data/branchSelectors.js';
import { loadDashboardData, validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { selectRecentRuns } from '../../src/data/selectors.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

test('normalizes only schema 2 with explicit modes and keeps branch/plugin source envelopes', () => {
  const source = createSchema2Publication();
  const { data, sourceData } = validatePublishedDashboardData(source);
  expect(data.schemaVersion).toBe(2);
  expect(data.runs).toHaveLength(24);
  expect(data.allRuns).toHaveLength(44);
  expect(data.pluginRuns).toHaveLength(45);
  expect(data.runs.every(({ branch, plugin }) => branch === 'develop' && plugin.id === 'vanilla')).toBe(true);
  expect(data.modes).toEqual(['ST', 'MT']);
  expect(data.latestCommitRun.runId).toBe('fictional-develop-23');
  expect(data.latestRun.runId).toBe('fictional-develop-21');
  expect(data.runs[0].tests[0].testId).toBe('gfx1250:ST:a');
  expect(data.allRuns.at(-1).sourceBase).toBeUndefined();
  expect(sourceData).toEqual(source);
  expect(data.runs.find(({ runId }) => runId === 'fictional-develop-14').tests[0].durationSeconds).toBe(0);
});

test.each([123, true, false, ['attempt-1']].map((id) => ({ id })))('rejects non-string attempt ID $id before normalization', ({ id }) => {
  const source = createSchema2Publication();
  source.runs = [{ ...source.runs[0], id }];
  source.index.runFiles = [`runs/${id}.json`];
  expect(() => validatePublishedDashboardData(source)).toThrow(/does not match the schema-2 run contract/);
});

test('schema 1 reports an explicit migration error', () => {
  const source = createSchema2Publication();
  source.metadata.schemaVersion = 1;
  expect(() => validatePublishedDashboardData(source)).toThrow(/schema.?1.*migrat/i);
});

test('empty and branch-only snapshots remain valid', () => {
  const source = createSchema2Publication();
  source.runs = []; source.index.runFiles = [];
  const { data } = validatePublishedDashboardData(source);
  expect(data.runs).toEqual([]);
  expect(data.latestRun).toBeNull();
  const branches = createSchema2Publication();
  branches.runs = branches.runs.filter(({ source }) => source.branch !== 'develop');
  branches.index.runFiles = branches.runs.map(({ id }) => `runs/${id}.json`);
  expect(validatePublishedDashboardData(branches).data.allRuns).toHaveLength(20);
});

test('normalized reload validates result identities, statuses and duplicate attempts dynamically', () => {
  const data = validatePublishedDashboardData(createSchema2Publication()).data;
  const invalidResult = structuredClone(data);
  invalidResult.pluginRuns[0].tests[0].durationSeconds = -1;
  expect(() => loadDashboardData(invalidResult)).toThrow(/durationSeconds/);
  const invalidIdentity = structuredClone(data);
  invalidIdentity.pluginRuns[0].tests[0].testId = 'gfx1250:MT:a';
  expect(() => loadDashboardData(invalidIdentity)).toThrow(/identity/);
  const duplicate = structuredClone(data);
  duplicate.pluginRuns.push(duplicate.pluginRuns[0]);
  expect(() => loadDashboardData(duplicate)).toThrow(/Duplicate run ID/);
});

test('configuration targets must be declared strings, not coerced numbers', () => {
  const source = createSchema2Publication();
  const run = source.runs[0];
  run.configurations = [run.configurations[0]];
  run.configurations[0].target = 123;
  source.catalogs[run.testCatalog].configurations = { '123:ST': ['a', 'b', 'c'] };
  source.runs = [run]; source.index.runFiles = [`runs/${run.id}.json`];
  expect(() => validatePublishedDashboardData(source)).toThrow(/configuration/);
});

test.each([
  { mode: ['ST'], threadCount: 1 },
  { mode: ['ST'], threadCount: 8 },
  { mode: ['MT'], threadCount: 8 },
])('DATA-01 rejects coerced array mode $mode with $threadCount threads', ({ mode, threadCount }) => {
  const source = createSchema2Publication();
  const run = source.runs[0];
  source.runs = [run]; source.index.runFiles = [`runs/${run.id}.json`];
  run.configurations = [run.configurations[0]];
  Object.assign(run.configurations[0], { mode, threadCount });
  expect(() => validatePublishedDashboardData(source)).toThrow(/invalid.*configuration/i);
});

test('DATA-02 result extensions cannot replace authoritative catalog definitions', () => {
  const source = createSchema2Publication();
  const run = source.runs[0];
  source.runs = [run]; source.index.runFiles = [`runs/${run.id}.json`];
  const result = run.configurations[0].results[0];
  const definition = source.catalogs[run.testCatalog].tests.find(({ id }) => id === result.testId);
  Object.assign(result, { id: 'shadow-id', suite: 'not-a-catalog-suite', name: 'shadow name', problem: { size: 'shadow' } });
  const { data, sourceData } = validatePublishedDashboardData(source);
  expect(data.runs[0].tests[0]).toMatchObject(definition);
  expect(sourceData.runs[0].configurations[0].results[0]).toEqual(result);
});

test('DATA-02 result suite extensions cannot hide a failed workload from coverage', () => {
  const source = createSchema2Publication();
  const run = source.runs.find(({ id }) => id === 'fictional-develop-08');
  source.runs = [run]; source.index.runFiles = [`runs/${run.id}.json`];
  run.configurations[0].results.find(({ testId }) => testId === 'd').suite = 'not-a-catalog-suite';
  const { data } = validatePublishedDashboardData(source);
  const recent = selectRecentRuns(data, { targets: ['gfx1250'], modes: ['ST'], suites: data.suites });
  expect(recent[0]).toMatchObject({ failed: 1, timeout: 1, completed: 2, total: 4, duration: null });
});

test.each(['absent map', 'empty map', 'missing referenced envelope'])('DATA-04 rejects normalized reload with %s', (missing) => {
  const source = createSchema2Publication();
  source.runs = [source.runs[0]];
  source.index.runFiles = source.runs.map(({ id }) => `runs/${id}.json`);
  const normalized = structuredClone(validatePublishedDashboardData(source).data);
  if (missing === 'absent map') delete normalized.catalogs;
  else if (missing === 'empty map') normalized.catalogs = {};
  else delete normalized.catalogs[normalized.pluginRuns[0].testCatalog];
  expect(() => loadDashboardData(normalized)).toThrow(/catalog/i);
});

test('DATA-04 missing catalogs cannot conceal omitted required normalized workloads', () => {
  const source = createSchema2Publication();
  source.runs = [source.runs[0]];
  source.index.runFiles = source.runs.map(({ id }) => `runs/${id}.json`);
  const normalized = structuredClone(validatePublishedDashboardData(source).data);
  delete normalized.catalogs;
  normalized.pluginRuns[0].tests = normalized.pluginRuns[0].tests.filter(({ logicalTestId }) => logicalTestId !== 'c');
  expect(() => loadDashboardData(normalized)).toThrow(/catalog/i);
});

test('DATA-04 normalized JSON round trips retain authoritative catalog membership', () => {
  const data = validatePublishedDashboardData(createSchema2Publication()).data;
  const serialized = JSON.parse(JSON.stringify({ ...data, backfillRunIds: [...data.backfillRunIds] }));
  expect(loadDashboardData(serialized)).toEqual(data);
  serialized.pluginRuns[0].tests = serialized.pluginRuns[0].tests.filter(({ logicalTestId }) => logicalTestId !== 'c');
  expect(() => loadDashboardData(serialized)).toThrow(/exactly one result.*catalog workload/i);
});

test.each(['publication', 'normalized reload'])('DATA-06 rejects array source.commit SHA via %s', (entrypoint) => {
  const source = createSchema2Publication();
  const run = source.runs.find(({ id }) => id === 'fictional-branch-01');
  source.runs = [run]; source.index.runFiles = [`runs/${run.id}.json`];
  const input = entrypoint === 'publication' ? source
    : JSON.parse(JSON.stringify(validatePublishedDashboardData(source).data));
  const identity = entrypoint === 'publication' ? run.source : input.pluginRuns[0].provenance;
  const key = entrypoint === 'publication' ? 'commit' : 'rocjitsuCommitSha';
  identity[key] = [identity[key]];
  const validate = entrypoint === 'publication' ? validatePublishedDashboardData : loadDashboardData;
  expect(() => validate(input)).toThrow(/Run fictional-branch-01 does not match the schema-2 run contract/);
});

test.each(['publication', 'normalized reload'])('DATA-06 rejects array source.base.commit SHA via %s', (entrypoint) => {
  const source = createSchema2Publication();
  const run = source.runs.find(({ id }) => id === 'fictional-branch-01');
  source.runs = [run]; source.index.runFiles = [`runs/${run.id}.json`];
  const input = entrypoint === 'publication' ? source
    : JSON.parse(JSON.stringify(validatePublishedDashboardData(source).data));
  const base = entrypoint === 'publication' ? run.source.base : input.pluginRuns[0].sourceBase;
  base.commit = [base.commit];
  const validate = entrypoint === 'publication' ? validatePublishedDashboardData : loadDashboardData;
  expect(() => validate(input)).toThrow(/Run fictional-branch-01 has an invalid source base/);
});

test.each(['publication', 'normalized reload'])('DATA-06 preserves primitive full SHAs for branch search and exact base via %s', (entrypoint) => {
  const source = createSchema2Publication();
  source.runs = source.runs.filter(({ id }) => ['fictional-branch-01', 'fictional-develop-20', 'fictional-develop-21'].includes(id));
  source.index.runFiles = source.runs.map(({ id }) => `runs/${id}.json`);
  const branch = source.runs.find(({ id }) => id === 'fictional-branch-01');
  branch.source.commit = branch.source.commit.toUpperCase();
  const normalized = validatePublishedDashboardData(source).data;
  const data = entrypoint === 'publication' ? normalized : loadDashboardData(JSON.parse(JSON.stringify(normalized)));
  const candidate = data.allRuns.find(({ runId }) => runId === branch.id);
  expect(candidate.provenance.rocjitsuCommitSha).toBe(branch.source.commit);
  expect(candidate.sourceBase).toEqual(branch.source.base);
  expect(selectPublishedBranches(data, { query: branch.source.commit.toLowerCase() }).map(({ latestRun }) => latestRun.runId)).toEqual([branch.id]);
  expect(selectPublishedBranches(data, { query: 'nonmatching-query' })).toEqual([]);
  expect(selectAutomaticReference(data, candidate)).toMatchObject({ reason: 'exact-base', run: { runId: 'fictional-develop-20' } });
});

test('equivalent plugin source metadata does not depend on JSON property order', () => {
  const source = createSchema2Publication();
  const vanilla = source.runs.find(({ id }) => id === 'fictional-develop-23');
  const plugin = source.runs.at(-1);
  vanilla.source.base = { branch: 'develop', commit: source.runs[0].source.commit };
  plugin.source.base = { commit: source.runs[0].source.commit, branch: 'develop' };
  expect(() => validatePublishedDashboardData(source)).not.toThrow();
});
