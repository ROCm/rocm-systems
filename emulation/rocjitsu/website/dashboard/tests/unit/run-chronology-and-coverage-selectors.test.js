import { expect, test } from 'vitest';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { compareRunExecution, compareCommitPosition, sortRunsByCommit } from '../../src/data/runOrdering.js';
import { selectBenchmarkSeries, periodKey } from '../../src/data/selectors.js';
import { selectOverviewProjection } from '../../src/data/overviewProjection.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
const data = adaptPublishedDashboardData(createMaintainedPublication()).data;
const filters = { targets: ['gfx1250'], suites: data.suites, modes: ['single'] };

// Keep chronology/coverage behaviors, without tying tests to historical production-like SHAs.
test('late backfill is recent by execution but remains in its source commit position', () => {
  const backfill = data.latestRun;
  expect(backfill.runId).toBe('fictional-develop-21');
  expect(data.latestCommitRun.runId).toBe('fictional-develop-23');
  expect(compareCommitPosition(backfill, data.latestCommitRun)).toBeLessThan(0);
  expect(compareRunExecution(backfill, data.latestCommitRun)).toBeGreaterThan(0);
  expect(sortRunsByCommit(data.runs).at(-1)).toBe(data.latestCommitRun);
  expect(selectBenchmarkSeries(data, filters, 'a').runs.at(-1)).toBe(data.latestCommitRun);
  expect(selectOverviewProjection(data, filters, {range: '1D'}).anchorDay).toBe(data.latestCommitRun.commitTimestamp.slice(0, 10));
});

test('reruns keep both identities adjacent without changing commit chronology', () => {
  const original = data.runs[0]; const retry = { ...original, runId: 'retry', timestamp: data.generatedAt };
  const input = { ...data, runs: [...data.runs, retry] };
  const result = selectBenchmarkSeries(input, filters, 'a');
  expect(result.runs.slice(0, 2)).toEqual([original, retry]);
  expect(result.labels[0]).toContain('1/2');
  expect(result.labels[1]).toContain('2/2');
});

test('weekly gaps occupy day bands and history sufficiency uses calendar coverage', () => {
  const history = selectOverviewProjection(data, filters, {range: '1W'});
  expect(history.dayKeys).toHaveLength(7);
  expect(history.slots.every((slot) => slot.x >= 0 && slot.x < 7)).toBe(true);
  expect(periodKey('2026-10-04T01:00:00Z')).toBe('2026-09-28');
  expect(periodKey('2026-10-04T01:00:00Z', 'monthly')).toBe('2026-10');
});

test.each([
  { label: 'lower offset', commit: '0000-01-01T00:30:00+01:00', retry: '0000-01-01T00:40:00+01:00', next: '0000-01-01T00:00:00Z',
    finished: '0000-01-01T00:10:00Z', day: '-000001-12-31', month: '-000001-12', monday: '-000001-12-27', nextDay: '0000-01-01', nextMonday: '-000001-12-27' },
  { label: 'upper offset', commit: '9999-12-31T23:30:00-01:00', retry: '9999-12-31T23:40:00-01:00', next: '9999-12-31T23:50:00-01:00',
    finished: '9999-12-31T23:55:00-01:00', day: '+010000-01-01', month: '+010000-01', monday: '9999-12-27', nextDay: '+010000-01-01', nextMonday: '9999-12-27' },
])('validated benchmark UTC period identities and labels preserve $label measurements and reruns', (input) => {
  const publication = createMaintainedPublication();
  publication.runs = publication.runs.slice(0, 3);
  const sha = publication.runs[0].provenance.rocjitsu.rocjitsu_commit_sha;
  publication.runs.forEach((run, index) => {
    const committed = index === 2 ? input.next : input.commit;
    run.provenance.rocjitsu.rocjitsu_commit_sha = index === 2 ? '2'.repeat(40) : sha;
    run.provenance.rocjitsu.rocjitsu_commit_timestamp = committed;
    run.provenance.corpus.corpus_commit_timestamp = committed;
    run.execution_summary.timestamp = committed;
    run.execution_summary.finished_at = index === 0 ? committed : index === 1 ? input.retry : input.finished;
  });
  publication.runs.reverse();
  publication.index = { generatedAt: input.finished,
    runFiles: publication.runs.map((run) => `runs/default-branch/${run.publication.id}.json`) };
  const originalPublication = JSON.stringify(publication);
  const { data: normalized } = adaptPublishedDashboardData(publication);
  const originalData = JSON.stringify(normalized);
  const chosen = { targets: ['gfx1250'], suites: ['Triton'], modes: ['single'] };
  const result = selectBenchmarkSeries(normalized, chosen, 'a');
  expect(result.runs.map((run) => run.runId)).toEqual(['fictional-develop-00', 'fictional-develop-01', 'fictional-develop-02']);
  expect(result.runs.every((run) => normalized.runs.includes(run))).toBe(true);
  expect(result.series).toHaveLength(1);
  expect(result.series[0].points.map((point) => point.value)).toEqual([20, 19.8, 19.6]);
  result.series[0].records.forEach((record, index) => {
    expect(record.run).toBe(result.runs[index]);
    expect(record.test).toBe(record.run.tests.find((test) => test.logicalTestId === 'a' && test.target === 'gfx1250' && test.mode === 'single'));
    expect(result.series[0].points[index].record).toBe(record);
  });
  // Soft assertions still fail the case; they expose all three truncated branches in RED.
  expect.soft(periodKey(input.commit, 'daily')).toBe(input.day);
  expect.soft(periodKey(input.commit, 'monthly')).toBe(input.month);
  expect.soft(periodKey(input.commit, 'weekly')).toBe(input.monday);
  expect.soft(periodKey(input.commit)).toBe(input.monday);
  expect.soft(periodKey(input.commit, 'unknown')).toBe(input.monday);
  expect.soft(periodKey(input.next)).toBe(input.nextMonday);
  const date = new Date(input.commit);
  const time = date.getTime();
  expect.soft(periodKey(date, 'daily')).toBe(input.day);
  expect(date.getTime()).toBe(time);
  const expectedDay = new Date(`${input.day}T12:00:00Z`).toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' });
  const expectedNextDay = new Date(`${input.nextDay}T12:00:00Z`).toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' });
  expect.soft(result.labels).toEqual([`${expectedDay}\n${sha.slice(0, 8)} · 1/2`, `${expectedDay}\n${sha.slice(0, 8)} · 2/2`, `${expectedNextDay}\n22222222`]);
  expect(result.runs.map((run) => run.commitTimestamp)).toEqual([input.commit, input.commit, input.next]);
  expect(JSON.stringify(publication)).toBe(originalPublication);
  expect(JSON.stringify(normalized)).toBe(originalData);
  console.info('benchmark-period-proof', JSON.stringify({ boundary: input.label,
    daily: periodKey(input.commit, 'daily'), monthly: periodKey(input.commit, 'monthly'), monday: periodKey(input.commit),
    labels: result.labels, expectedLabels: [`${expectedDay}\n${sha.slice(0, 8)} · 1/2`, `${expectedDay}\n${sha.slice(0, 8)} · 2/2`, `${expectedNextDay}\n22222222`],
    runs: result.runs.map((run) => run.runId), values: result.series[0].points.map((point) => point.value), sourceIdentityAndImmutabilityChecked: true }));
});

test('ordinary period keys retain Date input and default or unknown weekly fallback', () => {
  const date = new Date('2026-10-04T01:00:00Z');
  expect(periodKey(date, 'daily')).toBe('2026-10-04');
  expect(periodKey(date, 'monthly')).toBe('2026-10');
  expect(periodKey(date)).toBe('2026-09-28');
  expect(periodKey(date, 'unknown')).toBe('2026-09-28');
  expect(date.toISOString()).toBe('2026-10-04T01:00:00.000Z');
});
