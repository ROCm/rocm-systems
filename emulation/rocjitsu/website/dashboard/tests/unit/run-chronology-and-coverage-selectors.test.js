import { expect, test } from 'vitest';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { compareRunExecution, compareCommitPosition, sortRunsByCommit } from '../../src/data/runOrdering.js';
import { previousCompletedRunForFilters, selectAggregateRunSeries, selectBenchmarkSeries, selectOverview, selectRunReliability, selectFailures, periodKey } from '../../src/data/selectors.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
const data = validatePublishedDashboardData(createSchema2Publication()).data;
const filters = { targets: ['gfx1250'], suites: data.suites, modes: ['ST'] };

// Keep chronology/coverage behaviors, without tying tests to historical production-like SHAs.
test('late backfill is recent by execution but remains in its source commit position', () => {
  const backfill = data.latestRun;
  expect(backfill.runId).toBe('fictional-develop-21');
  expect(data.latestCommitRun.runId).toBe('fictional-develop-23');
  expect(compareCommitPosition(backfill, data.latestCommitRun)).toBeLessThan(0);
  expect(compareRunExecution(backfill, data.latestCommitRun)).toBeGreaterThan(0);
  expect(sortRunsByCommit(data.runs).at(-1)).toBe(data.latestCommitRun);
  expect(selectBenchmarkSeries(data, filters, 'a').runs.at(-1)).toBe(data.latestCommitRun);
  expect(selectAggregateRunSeries(data, filters).runs.at(-1)).toBe(data.latestCommitRun);
  expect(selectOverview(data, filters, '1D').history.anchorDay).toBe(data.latestCommitRun.commitTimestamp.slice(0, 10));
});

test('reruns keep both identities adjacent without changing commit chronology', () => {
  const original = data.runs[0]; const retry = { ...original, runId: 'retry', timestamp: data.generatedAt };
  const input = { ...data, runs: [...data.runs, retry] };
  const result = selectBenchmarkSeries(input, filters, 'a');
  expect(result.runs.slice(0, 2)).toEqual([original, retry]);
  expect(result.labels[0]).toContain('1/2');
  expect(result.labels[1]).toContain('2/2');
});

test('previous completion is independently selected for exact mode and suite', () => {
  const candidate = data.runs.find(({ runId }) => runId === 'fictional-develop-10');
  const baseline = previousCompletedRunForFilters(data.runs, candidate, filters);
  expect(baseline).toBeNull(); // Older catalogs cannot supply every current workload.
  expect(previousCompletedRunForFilters(data.runs, candidate, { ...filters, modes: ['MT'], suites: ['Triton'] }).runId).toBe('fictional-develop-09');
});

test('aggregate explorer exposes configuration-specific catalog boundaries and gaps', () => {
  const result = selectAggregateRunSeries(data, { ...filters, modes: ['ST', 'MT'] });
  expect(result.series).toHaveLength(2);
  expect(result.series[0].catalogBreaks).toEqual([8]);
  expect(result.series[0].data[8].value).toBeNull();
  expect(result.series[1].data[8].value).toBeNull();
  expect(result.series[0].data[14].value).toBeTypeOf('number');
});

test('weekly gaps occupy day bands and history sufficiency uses calendar coverage', () => {
  const { history } = selectOverview(data, filters, '1W');
  expect(history.dayKeys).toHaveLength(7);
  expect(history.slots.every((slot) => slot.x >= 0 && slot.x < 7)).toBe(true);
  expect(selectOverview(data, filters, '3M').history.insufficientData).toBe(true);
  expect(periodKey('2026-10-04T01:00:00Z')).toBe('2026-09-28');
  expect(periodKey('2026-10-04T01:00:00Z', 'monthly')).toBe('2026-10');
});

test('failure and timeout counts remain separate from missing-mode availability', () => {
  const result = selectRunReliability(data, { ...filters, targets: data.targets, modes: ['ST', 'MT'] }, 24);
  expect(result.failed).toBeGreaterThan(0);
  expect(result.timeout).toBeGreaterThan(0);
  expect(result.issueRuns.some(({ run }) => run.runId === 'fictional-develop-12')).toBe(false);
  expect(selectFailures(data, filters).every(({ test }) => test.mode === 'ST')).toBe(true);
});
