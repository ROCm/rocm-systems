import { expect, test } from 'vitest';
import { selectAutomaticReference, selectPublishedBranches } from '../../src/data/branchSelectors.js';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { fixtureRunPath } from '../fixtures/runPath.js';

test('flat corpus runs retain normalized view values, literal policies and exact source envelopes', () => {
  const source = createMaintainedPublication();
  const before = structuredClone(source);
  const { data, sourceData } = adaptPublishedDashboardData(source);
  expect(data.runs).toHaveLength(24);
  expect(data.allRuns).toHaveLength(44);
  expect(data.modes).toEqual(['single', 'default']);
  expect(data.latestCommitRun.runId).toBe('fictional-develop-23');
  expect(data.latestRun.runId).toBe('fictional-develop-21');
  expect(data.runs[0].tests[0].testId).toBe('gfx1250/single:a');
  expect(data.runs.find(({ runId }) => runId === 'fictional-develop-14').tests[0].durationSeconds).toBe(source.runs[14].benchmark_results[0].timing_results_s[0]);
  expect(sourceData).toEqual(source);
  expect(source).toEqual(before);
});

test.each([123, true, false, ['attempt-1']])('rejects unusable publication ID %j', (id) => {
  const source = createMaintainedPublication();
  source.runs[0].publication.id = id;
  expect(() => adaptPublishedDashboardData(source)).toThrow(/publication ID/);
});

test('empty and branch-only snapshots remain usable', () => {
  const source = createMaintainedPublication();
  source.runs = source.runs.filter(({ publication }) => publication.source.branch !== 'develop');
  source.index.runFiles = source.runs.map(fixtureRunPath);
  expect(adaptPublishedDashboardData(source).data.allRuns).toHaveLength(20);
  source.runs = []; source.index.runFiles = [];
  expect(adaptPublishedDashboardData(source).data.latestRun).toBeNull();
});

test('branch search, exact base and fallback reference keep their existing meaning', () => {
  const source = createMaintainedPublication();
  const { data } = adaptPublishedDashboardData(source);
  const branch = data.allRuns.find(({ runId }) => runId === 'fictional-branch-01');
  expect(selectPublishedBranches(data, { query: branch.provenance.rocjitsuCommitSha }).map(({ latestRun }) => latestRun.runId)).toEqual([branch.runId]);
  expect(selectAutomaticReference(data, branch)).toMatchObject({ reason: 'exact-base', run: { runId: 'fictional-develop-20' } });
  expect(selectAutomaticReference(data, { ...branch, sourceBase: undefined })).toMatchObject({ reason: 'earlier-develop', run: { runId: 'fictional-develop-21' } });
});
