import { fixtureRunPath } from '../fixtures/runPath.js';
import { expect, test } from 'vitest';
import { selectAutomaticReference, selectPublishedBranches } from '../../src/data/branchSelectors.js';
import * as dashboardValidation from '../../src/data/dashboardValidation.js';
import * as dashboardData from '../../src/data/dashboardData.js';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { selectRecentRunSummaries } from '../../src/data/selectors.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

test('data modules expose published loading rather than normalized JSON re-import', () => {
  expect(dashboardValidation).not.toHaveProperty('loadDashboardData');
  expect(dashboardData).not.toHaveProperty('loadDashboardData');
});

test('normalizes current runs with explicit modes and keeps branch source envelopes', () => {
  const source = createSchema2Publication();
  const { data, sourceData } = validatePublishedDashboardData(source);
  expect(data.schemaVersion).toBe(2);
  expect(sourceData.index).not.toHaveProperty('schemaVersion');
  for (const catalog of Object.values(sourceData.catalogs)) expect(catalog).not.toHaveProperty('schemaVersion');
  for (const run of sourceData.runs) expect(run.schemaVersion).toBe(2);
  expect(data.runs).toHaveLength(24);
  expect(data.allRuns).toHaveLength(44);
  expect(data).not.toHaveProperty('pluginRuns');
  expect(data.runs.every(({ branch }) => branch === 'develop')).toBe(true);
  expect(data.modes).toEqual(['ST', 'MT']);
  expect(data.latestCommitRun.runId).toBe('fictional-develop-23');
  expect(data.latestRun.runId).toBe('fictional-develop-21');
  expect(data.runs[0].tests[0].testId).toBe('gfx1250:ST:a');
  expect(data.allRuns.at(-1).sourceBase).toBeUndefined();
  expect(sourceData).toEqual(source);
  for (const run of [...sourceData.runs, ...data.allRuns]) {
    expect(run).not.toHaveProperty('plugin');
    expect(run).not.toHaveProperty('comparisonId');
    expect(run).not.toHaveProperty('workflowUrl');
    if (run.execution) expect(run.execution).not.toHaveProperty('workflowUrl');
    for (const configuration of run.configurations) {
      expect(['ST', 'MT']).toContain(configuration.threadingMode);
      expect(configuration).not.toHaveProperty('mode');
      expect(configuration).not.toHaveProperty('threadCount');
    }
  }
  expect(data.runs.find(({ runId }) => runId === 'fictional-develop-14').tests[0].durationSeconds).toBe(0);
});

test.each([123, true, false, ['attempt-1']].map((id) => ({ id })))('rejects non-string attempt ID $id before normalization', ({ id }) => {
  const source = createSchema2Publication();
  source.runs = [{ ...source.runs[0], id }];
  source.index.runFiles = [`runs/default-branch/${id}.json`];
  expect(() => validatePublishedDashboardData(source)).toThrow(/does not match the schema-2 run contract/);
});

test('unsupported run version reports an explicit schema error', () => {
  const source = createSchema2Publication();
  source.runs[0].schemaVersion = 99;
  expect(() => validatePublishedDashboardData(source)).toThrow(/schema.*99/i);
});

test('empty and branch-only snapshots remain valid', () => {
  const source = createSchema2Publication();
  source.runs = []; source.index.runFiles = [];
  const { data } = validatePublishedDashboardData(source);
  expect(data.runs).toEqual([]);
  expect(data.latestRun).toBeNull();
  const branches = createSchema2Publication();
  branches.runs = branches.runs.filter(({ source }) => source.branch !== 'develop');
  branches.index.runFiles = branches.runs.map(fixtureRunPath);
  expect(validatePublishedDashboardData(branches).data.allRuns).toHaveLength(20);
});

test('configuration targets must be declared strings, not coerced numbers', () => {
  const source = createSchema2Publication();
  const run = source.runs[0];
  run.configurations = [run.configurations[0]];
  run.configurations[0].target = 123;
  source.catalogs[run.testCatalog].configurations = { '123:ST': ['a', 'b', 'c'] };
  source.runs = [run]; source.index.runFiles = [fixtureRunPath(run)];
  expect(() => validatePublishedDashboardData(source)).toThrow(/configuration/);
});

test.each([
  { threadingMode: ['ST'] },
  { threadingMode: ['MT'] },
])('DATA-01 rejects coerced array threadingMode $threadingMode', ({ threadingMode }) => {
  const source = createSchema2Publication();
  const run = source.runs[0];
  source.runs = [run]; source.index.runFiles = [fixtureRunPath(run)];
  run.configurations = [run.configurations[0]];
  Object.assign(run.configurations[0], { threadingMode });
  expect(() => validatePublishedDashboardData(source)).toThrow(/invalid.*configuration/i);
});

test('DATA-02 result extensions cannot replace authoritative catalog definitions', () => {
  const source = createSchema2Publication();
  const run = source.runs[0];
  source.runs = [run]; source.index.runFiles = [fixtureRunPath(run)];
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
  source.runs = [run]; source.index.runFiles = [fixtureRunPath(run)];
  run.configurations[0].results.find(({ testId }) => testId === 'd').suite = 'not-a-catalog-suite';
  const { data } = validatePublishedDashboardData(source);
  const recent = selectRecentRunSummaries(data.runs, { targets: ['gfx1250'], modes: ['ST'], suites: data.suites });
  expect(recent[0]).toMatchObject({ failed: 1, timeout: 1, completed: 2, total: 4, duration: null });
});

test('rejects array source.commit SHA in published input', () => {
  const source = createSchema2Publication();
  const run = source.runs.find(({ id }) => id === 'fictional-branch-01');
  source.runs = [run]; source.index.runFiles = [fixtureRunPath(run)];
  run.source.commit = [run.source.commit];
  expect(() => validatePublishedDashboardData(source)).toThrow(/Run fictional-branch-01 does not match the schema-2 run contract/);
});

test('rejects array source.base.commit SHA in published input', () => {
  const source = createSchema2Publication();
  const run = source.runs.find(({ id }) => id === 'fictional-branch-01');
  source.runs = [run]; source.index.runFiles = [fixtureRunPath(run)];
  run.source.base.commit = [run.source.base.commit];
  expect(() => validatePublishedDashboardData(source)).toThrow(/Run fictional-branch-01 has an invalid source base/);
});

test.each(['source.commit', 'source.base.commit'].flatMap((field) =>
  ['uppercase', 'mixed-case'].map((casing) => ({ field, casing }))))('rejects $casing $field before reference selection', ({ field, casing }) => {
  const source = createSchema2Publication();
  source.runs = source.runs.filter(({ id }) => ['fictional-branch-01', 'fictional-develop-20', 'fictional-develop-21'].includes(id));
  source.index.runFiles = source.runs.map(fixtureRunPath);
  const branch = source.runs.find(({ id }) => id === 'fictional-branch-01');
  const normalized = validatePublishedDashboardData(source).data;
  const candidate = normalized.allRuns.find(({ runId }) => runId === branch.id);
  expect(selectAutomaticReference(normalized, candidate)).toMatchObject({ reason: 'exact-base', run: { runId: 'fictional-develop-20' } });
  expect(selectAutomaticReference(normalized, { ...candidate, sourceBase: undefined })).toMatchObject({ reason: 'earlier-develop', run: { runId: 'fictional-develop-21' } });
  const identity = field === 'source.base.commit' ? branch.source.base : branch.source;
  identity.commit = casing === 'uppercase' ? identity.commit.toUpperCase() : identity.commit.replace(/[a-f]/, (letter) => letter.toUpperCase());
  const invalidSha = identity.commit;
  expect(() => validatePublishedDashboardData(source)).toThrow(field === 'source.commit' ? /schema-2 run contract/ : /invalid source base/);
  expect(identity.commit).toBe(invalidSha);
});

test('preserves canonical lowercase full SHAs for branch search and exact base', () => {
  const source = createSchema2Publication();
  source.runs = source.runs.filter(({ id }) => ['fictional-branch-01', 'fictional-develop-20', 'fictional-develop-21'].includes(id));
  source.index.runFiles = source.runs.map(fixtureRunPath);
  const branch = source.runs.find(({ id }) => id === 'fictional-branch-01');
  expect(branch.source.commit).toMatch(/^[0-9a-f]{40}$/);
  const { data } = validatePublishedDashboardData(source);
  const candidate = data.allRuns.find(({ runId }) => runId === branch.id);
  expect(candidate.provenance.rocjitsuCommitSha).toBe(branch.source.commit);
  expect(candidate.sourceBase).toEqual(branch.source.base);
  expect(selectPublishedBranches(data, { query: branch.source.commit.toLowerCase() }).map(({ latestRun }) => latestRun.runId)).toEqual([branch.id]);
  expect(selectPublishedBranches(data, { query: 'nonmatching-query' })).toEqual([]);
  expect(selectAutomaticReference(data, candidate)).toMatchObject({ reason: 'exact-base', run: { runId: 'fictional-develop-20' } });
});
