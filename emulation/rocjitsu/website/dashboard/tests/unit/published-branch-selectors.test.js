import { expect, test } from 'vitest';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
const data = validatePublishedDashboardData(createSchema2Publication()).data;
const selectors = await import('../../src/data/branchSelectors.js').catch(() => ({}));

test('published branches use the inclusive publication-relative 30-day window and searchable branch/PR/SHA', () => {
  expect(typeof selectors.selectPublishedBranches).toBe('function');
  const branches = selectors.selectPublishedBranches(data);
  expect(branches).toHaveLength(20);
  expect(branches[0].branch).toBe('fictional/optimization-20');
  expect(selectors.selectPublishedBranches(data, { pr: 'with-pr' })).toHaveLength(10);
  expect(selectors.selectPublishedBranches(data, { pr: 'no-pr' })).toHaveLength(10);
  expect(selectors.selectPublishedBranches(data, { query: '1000' })[0].branch).toBe('fictional/optimization-01');
  expect(selectors.selectPublishedBranches(data, { query: data.allRuns.at(-1).provenance.rocjitsuCommitSha })).toHaveLength(1);
  const edge = structuredClone(data.allRuns.at(-1));
  edge.branch = 'fictional/window-edge'; edge.timestamp = edge.commitTimestamp = '2026-09-05T12:00:00.000Z';
  const outside = { ...edge, runId: 'outside', branch: 'fictional/outside', timestamp: '2026-09-05T11:59:59.999Z', commitTimestamp: '2026-09-05T11:59:59.999Z' };
  const future = { ...edge, runId: 'future', branch: 'fictional/future', timestamp: '2026-10-05T12:00:00.001Z', commitTimestamp: '2026-10-05T12:00:00.001Z' };
  expect(selectors.selectPublishedBranches({ ...data, allRuns: [...data.allRuns, edge, outside, future] }).map(({ branch }) => branch)).toContain(edge.branch);
  expect(selectors.selectPublishedBranches({ ...data, allRuns: [outside, future] })).toEqual([]);
  const active = data.allRuns.at(-1);
  const previousAttempt = { ...outside, branch: active.branch, runId: 'old-active-attempt' };
  const activeGroup = selectors.selectPublishedBranches({ ...data, allRuns: [active, previousAttempt] });
  expect(activeGroup).toHaveLength(1);
  expect(activeGroup[0].runs).toEqual([active, previousAttempt]);
});

test('all qualifying branches are returned without a twenty-branch cap, newest executions first even for old commits', () => {
  const base = data.allRuns.find((run) => run.branch !== data.canonicalBranch && run.plugin.id === 'vanilla');
  const runs = Array.from({ length: 35 }, (_, index) => ({ ...base, runId: `run-${index}`, branch: `fictional/branch-${index}`, timestamp: new Date(Date.parse('2026-10-01T00:00:00Z') + index * 60000).toISOString(), commitTimestamp: '2026-09-01T00:00:00Z' }));
  const branches = selectors.selectPublishedBranches({ ...data, allRuns: runs });
  expect(branches).toHaveLength(35);
  expect(branches[0].branch).toBe('fictional/branch-34');
});

test('automatic reference prefers the latest exact develop-base attempt and otherwise uses strictly earlier execution', () => {
  expect(typeof selectors.selectAutomaticReference).toBe('function');
  const candidate = data.allRuns.find(({ runId }) => runId === 'fictional-branch-01');
  const base = data.runs.find(({ runId }) => runId === 'fictional-develop-20');
  const retry = { ...base, runId: 'retry-z', timestamp: candidate.timestamp };
  const tied = { ...retry, runId: 'retry-a' };
  expect(selectors.selectAutomaticReference({ ...data, runs: [...data.runs, retry, tied] }, candidate)).toMatchObject({ run: retry, reason: 'exact-base' });
  const fallback = { ...candidate, sourceBase: { branch: 'other', commit: candidate.provenance.rocjitsuCommitSha } };
  expect(selectors.selectAutomaticReference(data, fallback)).toMatchObject({ run: data.latestRun, reason: 'earlier-develop' });
  const beforeAll = { ...candidate, timestamp: data.runs[0].timestamp, sourceBase: undefined };
  expect(selectors.selectAutomaticReference(data, beforeAll)).toMatchObject({ run: null, reason: 'unavailable' });
  expect(selectors.selectAutomaticReference({ ...data, runs: [] }, candidate).reason).toBe('unavailable');
  expect(selectors.selectAutomaticReference(data, null).run).toBeNull();
});

test('configuration comparison is one exact pair and scope, including symmetric exclusions and seconds sorting', () => {
  expect(typeof selectors.selectConfigurationComparison).toBe('function');
  const candidate = data.allRuns.find(({ runId }) => runId === 'fictional-branch-03');
  const baseline = data.runs.find(({ runId }) => runId === 'fictional-develop-23');
  const result = selectors.selectConfigurationComparison(candidate, baseline, { target: 'gfx1250', mode: 'ST', suites: data.suites });
  expect(result.comparable).toHaveLength(3);
  expect(result.notComparable).toHaveLength(1);
  expect(result.comparable.every(({ candidateTest }) => candidateTest.target === 'gfx1250' && candidateTest.mode === 'ST')).toBe(true);
  const seconds = result.comparable.map(({ deltaSeconds }) => Math.abs(deltaSeconds));
  expect(seconds).toEqual([...seconds].sort((a, b) => b - a));
  const search = selectors.selectConfigurationComparison(candidate, baseline, { target: 'gfx1250', mode: 'ST', suites: data.suites, query: 'decode' });
  expect(search.comparable).toHaveLength(1);
  expect(search.notComparable).toHaveLength(0);
  const missing = data.allRuns.find(({ runId }) => runId === 'fictional-branch-02');
  const unavailable = selectors.selectConfigurationComparison(missing, baseline, { target: 'gfx1250', mode: 'MT', suites: data.suites });
  expect(unavailable.comparable).toHaveLength(0);
  expect(unavailable.notComparable).toHaveLength(4);
  expect(unavailable.candidateDuration).toBeNull();
});
