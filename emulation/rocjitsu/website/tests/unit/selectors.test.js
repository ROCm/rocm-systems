import { expect, test } from 'vitest';
import { loadDashboardData } from '../../src/data/dashboardData.js';
import { commitShaFor, isRunCompleted } from '../../src/data/runOrdering.js';
import {
  selectAggregateRunSeries,
  selectBenchmarkSeries,
  selectOverview,
  selectRecentRuns,
} from '../../src/data/selectors.js';
import { benchmarkData, cloneBenchmarkData } from '../fixtures/publishedData.js';

const gfx1250Filters = { targets: ['gfx1250'], suites: benchmarkData.suites };

test('counts failed tests in the run denominator', () => {
  const summary = selectRecentRuns(benchmarkData, gfx1250Filters, benchmarkData.runs.length)
    .find((candidate) => candidate.run.runId === 'benchmark-202606020530-187a7544');
  expect(summary).toMatchObject({ completed: 4, total: 5, failed: 1 });
  expect(isRunCompleted(summary.run)).toBe(false);
});

test('Overview uses the newest attempt of the newest commit even when it is incomplete', () => {
  const rawData = cloneBenchmarkData();
  const source = rawData.runs.find((run) => run.provenance.rocjitsuCommitSha.startsWith('31369c4d'));
  rawData.runs.push({
    ...source,
    runId: 'incomplete-latest-commit-attempt',
    timestamp: '2026-09-01T04:00:00.000Z',
    trigger: 'manual',
    tests: source.tests.map((result, index) => (index === 0 ? {
      ...result,
      durationSeconds: null,
      status: 'failed',
      exitCode: 1,
      error: 'Synthetic incomplete latest attempt',
    } : { ...result })),
  });
  const data = loadDashboardData(rawData);
  const overview = selectOverview(data, { targets: ['gfx1250'], suites: data.suites });

  expect(data.latestCommitRun.timestamp).toBe('2026-09-01T04:00:00.000Z');
  expect(commitShaFor(data.latestCommitRun)).toMatch(/^31369c4d/);
  expect(overview.metrics).toMatchObject({ completed: 6, total: 7, durationDelta: null });
  expect(overview.results.some((result) => result.status === 'failed')).toBe(true);
});

test('a newer rerun of the nearest earlier commit updates the Overview baseline', () => {
  const rawData = cloneBenchmarkData();
  const source = rawData.runs.find((run) => run.provenance.rocjitsuCommitSha.startsWith('9f774d29'));
  rawData.runs.push({
    ...source,
    runId: 'newer-9f774d29-rerun',
    timestamp: '2026-09-01T04:30:00.000Z',
    trigger: 'manual',
    tests: source.tests.map((result) => ({
      ...result,
      durationSeconds: Number.isFinite(result.durationSeconds)
        ? Number((result.durationSeconds * 2).toFixed(3))
        : result.durationSeconds,
    })),
  });
  const data = loadDashboardData(rawData);
  const overview = selectOverview(data, { targets: ['gfx1250'], suites: data.suites });
  const fp16Result = overview.results.find((result) => result.logicalTestId === 'triton-gemm-f16-1024');

  expect(commitShaFor(overview.candidate)).toMatch(/^31369c4d/);
  expect(overview.baseline.runId).toBe('newer-9f774d29-rerun');
  expect(fp16Result.previous.durationSeconds).toBeCloseTo(4.15, 2);
});

test('a late old-commit run stays out of Overview and remains addressable in Aggregate Explorer data', () => {
  const rawData = cloneBenchmarkData();
  const source = rawData.runs.find((run) => run.provenance.rocjitsuCommitSha.startsWith('31369c4d'));
  rawData.runs.push({
    ...source,
    runId: 'late-run-for-01d0c0de',
    timestamp: '2026-09-01T03:00:00.000Z',
    commitTimestamp: '2026-08-29T12:00:00.000Z',
    trigger: 'manual',
    provenance: {
      ...source.provenance,
      rocjitsuCommitSha: '01d0c0de00000000000000000000000000000000',
      commitMessage: 'Validate historical commit placement',
    },
  });
  const data = loadDashboardData(rawData);
  const filters = { targets: ['gfx1250'], suites: data.suites };
  const commitHistory = selectOverview(data, filters, '3M').history;
  const intradayHistory = selectOverview(data, filters, '1D').history;
  const recentRuns = selectRecentRuns(data, filters);
  const aggregate = selectAggregateRunSeries(data, filters);

  expect(commitHistory.slots.some((slot) => slot.run?.runId === 'late-run-for-01d0c0de')).toBe(false);
  expect(intradayHistory.slots.some((slot) => slot.run?.runId === 'late-run-for-01d0c0de')).toBe(false);
  expect(recentRuns[0]).toMatchObject({ olderCommit: true, run: { runId: 'late-run-for-01d0c0de' } });
  expect(aggregate.runs.some((run) => run.runId === 'late-run-for-01d0c0de')).toBe(true);
  expect(aggregate.runs.at(-1).runId).not.toBe('late-run-for-01d0c0de');
});

test('benchmark history keeps same-day commits as separate ordered points', () => {
  const labels = selectBenchmarkSeries(benchmarkData, gfx1250Filters, 'triton-gemm-f16-1024').labels;
  const latestLabels = labels.slice(-3);

  expect(new Set(labels).size).toBe(labels.length);
  expect(latestLabels.join(' ')).toContain('255eabe3');
  expect(latestLabels.join(' ')).toContain('9f774d29');
  expect(latestLabels.join(' ')).toContain('31369c4d');
});

test('a manual rerun of an older commit keeps its commit position and both attempts adjacent', () => {
  const series = selectBenchmarkSeries(benchmarkData, gfx1250Filters, 'triton-gemm-f16-1024');
  const backfillIndexes = series.labels
    .map((label, index) => (label.includes('8418072e') ? index : -1))
    .filter((index) => index >= 0);

  expect(backfillIndexes).toHaveLength(2);
  expect(backfillIndexes[1] - backfillIndexes[0]).toBe(1);
  expect(series.labels.slice(-3).join(' ')).not.toContain('8418072e');
  expect(selectRecentRuns(benchmarkData, gfx1250Filters)[0]).toMatchObject({
    latest: true,
    olderCommit: true,
  });
});
