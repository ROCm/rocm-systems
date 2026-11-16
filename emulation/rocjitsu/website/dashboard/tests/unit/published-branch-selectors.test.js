import { expect, test } from 'vitest';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import * as selectors from '../../src/data/branchSelectors.js';

const data = adaptPublishedDashboardData(createMaintainedPublication()).data;

test('published branches use the inclusive publication-relative 30-day window and searchable branch/PR/SHA', () => {
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
  const base = data.allRuns.find((run) => run.branch !== data.canonicalBranch);
  const runs = Array.from({ length: 35 }, (_, index) => ({ ...base, runId: `run-${index}`, branch: `fictional/branch-${index}`, timestamp: new Date(Date.parse('2026-10-01T00:00:00Z') + index * 60000).toISOString(), commitTimestamp: '2026-09-01T00:00:00Z' }));
  const branches = selectors.selectPublishedBranches({ ...data, allRuns: runs });
  expect(branches).toHaveLength(35);
  expect(branches[0].branch).toBe('fictional/branch-34');
});

const minutes = (from, offset) => new Date(Date.parse(from) + offset * 60_000).toISOString();
const withTimes = (run, runId, start, finish) => ({ ...run, runId, timestamp: start, finishedAt: finish });

test('automatic reference is the latest finish at or before the candidate start, never a run still running or killed', () => {
  const candidate = { ...data.allRuns.find(({ runId }) => runId === 'fictional-branch-01'), sourceBase: undefined };
  const base = data.runs.find(({ runId }) => runId === 'fictional-develop-20');
  const t0 = candidate.timestamp;
  const early = withTimes(base, 'finished-0900', minutes(t0, -80), minutes(t0, -50));   // finished before the candidate started
  const overlap = withTimes(base, 'overlap-1000', minutes(t0, -20), minutes(t0, 20));   // started earlier, finished after the candidate started
  const killed = withTimes(base, 'killed-1010', minutes(t0, -10), null);                // checkpoint with no finish record
  const pick = (runs) => selectors.selectAutomaticReference({ ...data, runs }, candidate);
  expect(pick([early, overlap, killed])).toMatchObject({ reason: 'earlier-develop', run: { runId: 'finished-0900' } });
  expect(pick([overlap, killed])).toMatchObject({ run: null, reason: 'unavailable' });
  // Finished runs rank by finish time, not start time: the later start finished first.
  const startsLater = withTimes(base, 'starts-later', minutes(t0, -70), minutes(t0, -60));
  const startsEarlier = withTimes(base, 'starts-earlier', minutes(t0, -90), minutes(t0, -30));
  expect(pick([startsLater, startsEarlier])).toMatchObject({ run: { runId: 'starts-earlier' } });
  // A run that finishes exactly when the candidate starts was complete in time.
  expect(pick([withTimes(base, 'boundary', minutes(t0, -30), t0)])).toMatchObject({ run: { runId: 'boundary' } });
});

test('with several develop attempts on the recorded base commit the latest finished one wins; a lone checkpoint is still used', () => {
  const candidate = data.allRuns.find(({ runId }) => runId === 'fictional-branch-01');
  const base = data.runs.find(({ runId }) => runId === 'fictional-develop-20');
  const t0 = candidate.timestamp;
  const finished = withTimes(base, 'finished-attempt', minutes(t0, -300), minutes(t0, -280));
  const timedOutRetry = withTimes(base, 'timed-out-retry', minutes(t0, -200), null);
  const pick = (runs) => selectors.selectAutomaticReference({ ...data, runs }, candidate);
  expect(pick([finished, timedOutRetry])).toMatchObject({ reason: 'exact-base', run: { runId: 'finished-attempt' } });
  expect(pick([timedOutRetry])).toMatchObject({ reason: 'exact-base', run: { runId: 'timed-out-retry' } });
  // Among finished attempts the one that finished last wins, even if another started later.
  const startsLaterFinishesFirst = withTimes(base, 'starts-later', minutes(t0, -290), minutes(t0, -285));   // starts after finished-attempt (-300) but finishes before it (-280)
  expect(pick([finished, startsLaterFinishesFirst])).toMatchObject({ reason: 'exact-base', run: { runId: 'finished-attempt' } });
});

test('automatic reference prefers the latest exact develop-base attempt and otherwise the latest finish at or before the candidate start', () => {
  const candidate = data.allRuns.find(({ runId }) => runId === 'fictional-branch-01');
  const base = data.runs.find(({ runId }) => runId === 'fictional-develop-20');
  const retry = { ...base, runId: 'retry-z', timestamp: candidate.timestamp };
  const tied = { ...retry, runId: 'retry-a' };
  expect(selectors.selectAutomaticReference({ ...data, runs: [...data.runs, retry, tied] }, candidate)).toMatchObject({ run: retry, reason: 'exact-base' });
  const fallback = { ...candidate, sourceBase: { branch: 'other', commit: candidate.provenance.rocjitsuCommitSha } };
  expect(selectors.selectAutomaticReference(data, fallback)).toMatchObject({ run: data.latestRun, reason: 'earlier-develop' });
  const beforeAll = { ...candidate, timestamp: new Date(Date.parse(data.runs[0].timestamp) - 1).toISOString(), sourceBase: undefined };
  expect(selectors.selectAutomaticReference(data, beforeAll)).toMatchObject({ run: null, reason: 'unavailable' });
  expect(selectors.selectAutomaticReference({ ...data, runs: [] }, candidate).reason).toBe('unavailable');
  expect(selectors.selectAutomaticReference(data, null).run).toBeNull();
});

test('configuration comparison is one exact pair and scope, including symmetric exclusions and seconds sorting', () => {
  const candidate = data.allRuns.find(({ runId }) => runId === 'fictional-branch-03');
  const baseline = data.runs.find(({ runId }) => runId === 'fictional-develop-23');
  const result = selectors.selectConfigurationComparison(candidate, baseline, { target: 'gfx1250', mode: 'single', suites: data.suites });
  expect(result.comparable).toHaveLength(3);
  expect(result.notComparable).toHaveLength(1);
  expect(result.comparable.every(({ candidateTest }) => candidateTest.target === 'gfx1250' && candidateTest.mode === 'single')).toBe(true);
  const seconds = result.comparable.map(({ deltaSeconds }) => Math.abs(deltaSeconds));
  expect(seconds).toEqual([...seconds].sort((a, b) => b - a));
  const search = selectors.selectConfigurationComparison(candidate, baseline, { target: 'gfx1250', mode: 'single', suites: data.suites, query: 'decode' });
  expect(search.comparable).toHaveLength(1);
  expect(search.notComparable).toHaveLength(0);
  const missing = data.allRuns.find(({ runId }) => runId === 'fictional-branch-02');
  const unavailable = selectors.selectConfigurationComparison(missing, baseline, { target: 'gfx1250', mode: 'default', suites: data.suites });
  expect(unavailable.comparable).toHaveLength(0);
  expect(unavailable.notComparable).toHaveLength(4);
  expect(unavailable.candidateDuration).toBeNull();
});
