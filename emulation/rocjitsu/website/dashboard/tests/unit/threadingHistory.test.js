import { expect, test } from 'vitest';
import { selectOverview, selectThreadingHistory } from '../../src/data/selectors.js';

const filters = { targets: ['gfx950'], suites: ['Triton'] };
function run(mode, date, duration, sha = date, logicalIds = ['matmul']) {
  return {
    runId: `${mode}-${date}-${sha}`,
    threadingMode: mode,
    timestamp: date,
    commitTimestamp: date,
    provenance: { rocjitsuCommitSha: sha },
    catalogId: `${mode}-catalog`,
    tests: logicalIds.map((logicalTestId) => ({
      testId: `${mode}-${logicalTestId}`,
      logicalTestId,
      target: 'gfx950', suite: 'Triton', status: 'completed', durationSeconds: duration,
    })),
  };
}
function section(mode, runs) {
  return { mode, data: { runs } };
}

test('same-day modes keep separate values, trimmed workloads, baselines and tooltip runs', () => {
  const defaults = [run('default', '2026-10-01T12:00:00Z', 10, 'a', ['a', 'b']), run('default', '2026-10-02T12:00:00Z', 9, 'b', ['a', 'b'])];
  const singles = [run('single', '2026-10-01T12:00:00Z', 100, 'a'), run('single', '2026-10-02T12:00:00Z', 120, 'b')];
  const history = selectThreadingHistory([section('default', defaults), section('single', singles)], filters);
  expect(history.slots).toHaveLength(2);
  expect(history.series.map(({ name }) => name)).toEqual(['gfx950 · Default', 'gfx950 · Single-thread']);
  expect(history.series[0]).toMatchObject({ data: [20, 18], baseline: 20, runs: defaults });
  expect(history.series[1]).toMatchObject({ data: [100, 120], baseline: 100, runs: singles });
  expect(history.modeSummaries.map(({ durationDelta }) => durationDelta)).toEqual([-10, 20]);
  expect(history.currentDuration).toBeNull();
});

test('daily histories share a calendar despite different starting dates', () => {
  const history = selectThreadingHistory([
    section('default', [run('default', '2026-10-01T12:00:00Z', 10)]),
    section('single', [run('single', '2026-10-03T12:00:00Z', 20)]),
  ], filters);
  expect(history.dayKeys).toEqual(['2026-10-01', '2026-10-02', '2026-10-03']);
  expect(history.series.map(({ data }) => data)).toEqual([[10, null, null], [null, null, 20]]);
  expect(history.slots.map(({ x }) => x)).toEqual([0, 1, 2]);
});

test.each(['1D', '1W'])('%s aligns commit slots, including simultaneous different commits', (range) => {
  const defaults = [run('default', '2026-10-07T12:00:00Z', 10, 'a'), run('default', '2026-10-07T13:00:00Z', 9, 'c')];
  const singles = [run('single', '2026-10-07T12:00:00Z', 20, 'b'), run('single', '2026-10-07T13:00:00Z', 18, 'c')];
  const history = selectThreadingHistory([section('default', defaults), section('single', singles)], filters, range);
  expect(history.slots).toHaveLength(3);
  expect(history.series.map(({ data }) => data)).toEqual([[10, null, 9], [null, 20, 18]]);
  expect(history.series[0].runs).toEqual([defaults[0], null, defaults[1]]);
  expect(history.series[1].runs).toEqual([null, singles[0], singles[1]]);
  expect(history.slots.map(({ x }) => x)).toEqual(range === '1D' ? [0, 1, 2] : [6.5, 6.5, 6 + 13 / 24]);
});

test('ranged histories use the latest selected commit day as a common anchor', () => {
  const history = selectThreadingHistory([
    section('default', [run('default', '2026-10-07T12:00:00Z', 10)]),
    section('single', [run('single', '2026-10-06T12:00:00Z', 20)]),
  ], filters, '1D');
  expect(history.anchorDay).toBe('2026-10-07');
  expect(history.series.map(({ data }) => data)).toEqual([[10], [null]]);
  expect(history.modeSummaries[0]).toMatchObject({ hasRuns: true, currentDuration: 10 });
  expect(history.modeSummaries[1]).toMatchObject({
    hasRuns: false, currentDuration: null, durationDelta: null, latestRun: null, firstRun: null,
  });
  const wider = selectThreadingHistory([
    section('default', [run('default', '2026-10-07T12:00:00Z', 10)]),
    section('single', [run('single', '2026-10-06T12:00:00Z', 20)]),
  ], filters, 'ALL');
  expect(wider.modeSummaries[1]).toMatchObject({ hasRuns: true, currentDuration: 20 });
});

test('insufficient coverage in one mode does not suppress the other mode', () => {
  const defaults = Array.from({ length: 30 }, (_, i) => run('default', `2026-09-${String(i + 1).padStart(2, '0')}T12:00:00Z`, 10));
  const history = selectThreadingHistory([
    section('default', defaults), section('single', [run('single', '2026-09-30T12:00:00Z', 20)]),
  ], filters, '1M');
  expect(history.insufficientData).toBe(false);
  expect(history.modeSummaries.map(({ insufficientData }) => insufficientData)).toEqual([false, true]);
  expect(history.series.map(({ insufficientData }) => insufficientData)).toEqual([false, true]);
  expect(history.series[0].data.every(Number.isFinite)).toBe(true);
});

test('empty modes and filters absent from a mode are safe', () => {
  expect(selectThreadingHistory([], filters).series).toEqual([]);
  const history = selectThreadingHistory([
    section('default', []), section('single', [run('single', '2026-10-07T12:00:00Z', 20)]),
  ], { ...filters, suites: ['Plugin'] });
  expect(history.series[0].data).toEqual([null]);
  for (const summary of history.modeSummaries) {
    expect(summary).toMatchObject({ hasRuns: false, currentDuration: null, latestRun: null });
  }
});


test('shared target selection preserves deltas for modes with disjoint targets', () => {
  const defaults = [run('default', '2026-10-01T12:00:00Z', 10), run('default', '2026-10-02T12:00:00Z', 9)]
    .map((item) => ({ ...item, tests: item.tests.map((result) => ({ ...result, target: 'gfx1250' })) }));
  const singles = [run('single', '2026-10-01T12:00:00Z', 100), run('single', '2026-10-02T12:00:00Z', 120)];
  const sharedFilters = { ...filters, targets: ['gfx1250', 'gfx950'] };
  const sections = [section('default', defaults), section('single', singles)];
  const combined = selectThreadingHistory(sections, sharedFilters);
  expect(combined.modeSummaries.map(({ durationDelta }) => durationDelta)).toEqual([-10, 20]);
  expect(combined.series.map(({ name }) => name)).toEqual(['gfx1250 · Default', 'gfx950 · Single-thread']);
  expect(sections.map(({ data }) => selectOverview(data, sharedFilters).metrics.durationDelta)).toEqual([-10, 20]);
  expect(selectOverview(sections[0].data, { ...sharedFilters, targets: [] }).history.series).toEqual([]);
});
