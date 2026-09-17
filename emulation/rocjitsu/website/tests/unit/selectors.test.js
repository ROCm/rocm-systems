import { expect, test } from 'vitest';
import { loadDashboardData } from '../../src/data/dashboardData.js';
import { commitShaFor, commitTimestampFor, sortRunsByCommit } from '../../src/data/runOrdering.js';
import {
  isRunCompletedForFilters,
  periodKey,
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
  expect(isRunCompletedForFilters(summary.run, gfx1250Filters)).toBe(false);
});

test('recent-run baselines follow the selected test scope', () => {
  const rowsFor = (filters) => selectRecentRuns(
    benchmarkData,
    filters,
    benchmarkData.runs.length,
  );
  const august30Row = (filters) => rowsFor(filters)
    .find(({ run }) => commitShaFor(run).startsWith('9398bd3f'));

  expect(commitShaFor(august30Row(gfx1250Filters).baseline)).toMatch(/^390ca630/);

  const tritonOnlyFilters = { ...gfx1250Filters, suites: ['Triton'] };
  expect(commitShaFor(august30Row(tritonOnlyFilters).baseline)).toMatch(/^0db03af1/);
});

test('history completion follows the selected target', () => {
  const mixedRun = benchmarkData.runs.find((run) => run.runId === 'benchmark-202608140530-f25f5a48');
  const gfx950Filters = { targets: ['gfx950'], suites: benchmarkData.suites };
  const gfx1250History = selectOverview(benchmarkData, gfx1250Filters, '3M').history;
  const gfx950History = selectOverview(benchmarkData, gfx950Filters, '3M').history;

  expect(isRunCompletedForFilters(mixedRun, gfx1250Filters)).toBe(true);
  expect(isRunCompletedForFilters(mixedRun, gfx950Filters)).toBe(false);
  expect(gfx1250History.slots.find((slot) => slot.dayKey === '2026-08-14').run?.runId)
    .toBe(mixedRun.runId);
  expect(gfx950History.slots.find((slot) => slot.dayKey === '2026-08-14').run).toBeNull();
});

test('1D baseline uses the previous commit day and falls back to its first point', () => {
  const oneDayHistory = selectOverview(benchmarkData, gfx1250Filters, '1D').history;
  const firstValue = oneDayHistory.series[0].data.find((value) => Number.isFinite(value));
  expect(oneDayHistory.series[0].baseline).toBeTypeOf('number');
  expect(oneDayHistory.series[0].baseline).not.toBe(firstValue);
  const previousDay = new Date(`${oneDayHistory.anchorDay}T12:00:00Z`);
  previousDay.setUTCDate(previousDay.getUTCDate() - 1);
  const previousDayKey = previousDay.toISOString().slice(0, 10);
  const previousDayRun = sortRunsByCommit(benchmarkData.runs.filter((run) => (
    periodKey(commitTimestampFor(run), 'daily') === previousDayKey
    && isRunCompletedForFilters(run, gfx1250Filters)
  ))).at(-1);
  const expectedBaseline = previousDayRun.tests
    .filter((test) => test.target === 'gfx1250' && gfx1250Filters.suites.includes(test.suite))
    .reduce((total, test) => total + test.durationSeconds, 0);
  expect(oneDayHistory.series[0].baseline).toBeCloseTo(expectedBaseline, 8);
  const candidateDuration = oneDayHistory.series.reduce(
    (total, series) => total + series.data.findLast((value) => Number.isFinite(value)),
    0,
  );
  const baselineDuration = oneDayHistory.series.reduce(
    (total, series) => total + series.baseline,
    0,
  );
  expect(oneDayHistory.durationDelta).toBeCloseTo(
    ((candidateDuration - baselineDuration) / baselineDuration) * 100,
    8,
  );

  const isolatedData = cloneBenchmarkData();
  isolatedData.runs = isolatedData.runs.filter((run) => (
    periodKey(commitTimestampFor(run), 'daily') !== previousDayKey
  ));
  const isolatedHistory = selectOverview(
    loadDashboardData(isolatedData),
    gfx1250Filters,
    '1D',
  ).history;
  const isolatedFirstValue = isolatedHistory.series[0].data.find((value) => Number.isFinite(value));
  expect(isolatedHistory.series[0].baseline).toBe(isolatedFirstValue);
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
  expect(fp16Result.previous.durationSeconds).toBeCloseTo(
    source.tests.find((test) => test.logicalTestId === 'triton-gemm-f16-1024').durationSeconds * 2,
    2,
  );
});

test('a late old-commit run remains visible in Overview and Aggregate Explorer data', () => {
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

  expect(commitHistory.slots.some((slot) => slot.run?.runId === 'late-run-for-01d0c0de')).toBe(true);
  expect(intradayHistory.slots.some((slot) => slot.run?.runId === 'late-run-for-01d0c0de')).toBe(false);
  expect(recentRuns[0]).toMatchObject({ olderCommit: true, run: { runId: 'late-run-for-01d0c0de' } });
  expect(aggregate.runs.some((run) => run.runId === 'late-run-for-01d0c0de')).toBe(true);
  expect(aggregate.runs.at(-1).runId).not.toBe('late-run-for-01d0c0de');
});

test('catalog changes break Aggregate and normalize Overview history to the latest workload', () => {
  const filters = { targets: ['gfx1250'], suites: benchmarkData.suites };
  const overview = selectOverview(benchmarkData, filters, 'ALL');
  const aggregate = selectAggregateRunSeries(benchmarkData, filters);
  const firstV1Index = overview.history.slots.findIndex((slot) => (
    slot.run?.catalogId === 'rocjitsu-core-v1'
  ));
  const oldCatalogIndex = overview.history.slots.findIndex((slot) => (
    slot.run?.runId === 'benchmark-202608130530-784750dd'
  ));
  const latestCatalogIndex = overview.history.slots.findIndex((slot) => (
    slot.run?.catalogId === 'rocjitsu-core-v2'
  ));
  const aggregateBreak = aggregate.series[0].catalogBreaks[0];

  expect(firstV1Index).toBeGreaterThanOrEqual(0);
  expect(oldCatalogIndex).toBeGreaterThanOrEqual(0);
  expect(latestCatalogIndex).toBeGreaterThan(firstV1Index);
  expect(overview.history.normalized).toBe(true);
  expect(overview.history.series[0].data[oldCatalogIndex]).toBeTypeOf('number');
  expect(overview.history.series[0].estimated[oldCatalogIndex]).toBe(true);
  expect(overview.history.series[0].estimatedTests[oldCatalogIndex]).toEqual(expect.arrayContaining([
    'tensile-gemm-fp8-4096',
    'deepseek-v3-decode-fp8',
  ]));
  expect(overview.history.series[0].data[oldCatalogIndex]).toBeCloseTo(1259.645, 3);
  expect(overview.history.series[0].estimated[latestCatalogIndex]).toBe(false);
  expect(aggregateBreak).toBeGreaterThan(0);
  expect(aggregate.runs[aggregateBreak].catalogId).not.toBe(
    aggregate.runs[aggregateBreak - 1].catalogId,
  );
});

test('weekly history keeps multiple commits per day and labels each day once', () => {
  const overview = selectOverview(benchmarkData, gfx1250Filters, '1W');
  const slotsByDay = new Map();
  overview.history.slots.forEach((slot) => {
    const slots = slotsByDay.get(slot.dayKey) ?? [];
    slots.push(slot);
    slotsByDay.set(slot.dayKey, slots);
  });

  expect(overview.history.slots).toHaveLength(56);
  expect(slotsByDay.get('2026-08-26').filter((slot) => slot.run)).toHaveLength(3);
  expect(slotsByDay.get('2026-08-30').filter((slot) => slot.run)).toHaveLength(2);
  expect(overview.history.slots.filter((slot) => slot.label).length).toBe(slotsByDay.size);
});

test('history sufficiency follows represented calendar days', () => {
  expect(selectOverview(benchmarkData, gfx1250Filters, '1M').history.insufficientData).toBe(false);
  const threeMonthHistory = selectOverview(benchmarkData, gfx1250Filters, '3M').history;
  const representedDays = new Set(
    threeMonthHistory.slots.filter((slot) => slot.run).map((slot) => slot.dayKey),
  ).size;
  expect(representedDays).toBe(69);
  expect(threeMonthHistory.insufficientData).toBe(false);
  expect(selectOverview(benchmarkData, gfx1250Filters, '6M').history.insufficientData).toBe(true);
  expect(selectOverview(benchmarkData, gfx1250Filters, 'YTD').history.insufficientData).toBe(true);
});

test('overview metric cards compare the latest commit with the oldest recorded commit', () => {
  const overview = selectOverview(benchmarkData, gfx1250Filters, 'ALL');
  const weekOverview = selectOverview(benchmarkData, gfx1250Filters, '1W');

  expect(commitShaFor(overview.candidate)).toMatch(/^31369c4d/);
  expect(commitShaFor(overview.metricsBaseline)).toMatch(/^86b362ea/);
  expect(overview.metrics.duration).toBeGreaterThan(0);
  expect(overview.metrics.durationDelta).toBeTypeOf('number');
  expect(commitShaFor(weekOverview.metricsBaseline)).toMatch(/^86b362ea/);
  expect(weekOverview.metrics.durationDelta).toBe(overview.metrics.durationDelta);
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
