import { expect, test } from 'vitest';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { selectRecentRunAttempts, selectRecentRunSummaries, selectBenchmarkSeries, selectBenchmarkCatalog, selectRunComparison } from '../../src/data/selectors.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';

const data = adaptPublishedDashboardData(createMaintainedPublication()).data;
const filters = { targets: ['gfx1250', 'gfx950'], suites: data.suites, modes: ['single', 'default'] };
const run = (id) => data.runs.find(({ runId }) => runId === `fictional-develop-${id}`);



test('policy-specific benchmark histories keep positive measurements and missing configurations as gaps', () => {
  const series = selectBenchmarkSeries(data, filters, 'a').series;
  expect(series.map(({ key, target, mode }) => ({ key, target, mode }))).toEqual([
    { key: 'gfx1250/single', target: 'gfx1250', mode: 'single' }, { key: 'gfx1250/default', target: 'gfx1250', mode: 'default' },
    { key: 'gfx950/single', target: 'gfx950', mode: 'single' }, { key: 'gfx950/default', target: 'gfx950', mode: 'default' },
  ]);
  expect(series[3].points[12]).toBeNull();
  expect(series[0].points[14].value).toBeGreaterThan(0);
  expect(selectBenchmarkSeries(data, { ...filters, modes: [] }, 'a').series).toEqual([]);
  expect(selectBenchmarkSeries(data, { ...filters, suites: [] }, 'a').series.every(({ points }) => points.every((p) => p === null))).toBe(true);
  expect(selectBenchmarkCatalog(data, { ...filters, modes: [] }).available).toEqual([]);
});

test('benchmark history retains first-match identity, scope defaults, attempt order and unavailable points without mutation', () => {
  const makeTest = (mode, status, durationSeconds) => ({ target: 'gfx1250', suite: 'Triton', logicalTestId: 'a', mode, status, durationSeconds });
  const first = makeTest('single', 'failed', null);
  const zero = makeTest('default', 'completed', 0);
  const timeout = makeTest('single', 'timeout', null);
  const nonfinite = makeTest('single', 'completed', Infinity);
  const measured = makeTest('default', 'completed', 2);
  const original = { runId: 'original', timestamp: '2026-01-01T12:00:00Z', commitTimestamp: '2026-01-01T00:00:00Z', provenance: { rocjitsuCommitSha: 'a'.repeat(40) }, tests: [
    { ...first, suite: 'Other' }, { ...first, suite: undefined }, { ...first, logicalTestId: 'other' },
    { ...first, target: 'gfx950' }, { ...first, mode: 'unknown' },
    first, makeTest('single', 'completed', 99), zero,
  ] };
  const retry = { ...original, runId: 'retry', timestamp: '2026-01-02T12:00:00Z', tests: [timeout] };
  const newer = { ...original, runId: 'newer', commitTimestamp: '2026-01-03T00:00:00Z', provenance: { rocjitsuCommitSha: 'b'.repeat(40) }, tests: [nonfinite, measured] };
  const empty = { ...newer, runId: 'empty', tests: [] };
  const omitted = { ...newer, runId: 'omitted', tests: undefined };
  const nullable = { ...newer, runId: 'nullable', tests: null };
  const input = { runs: [newer, retry, nullable, original, omitted, empty], testCatalog: [{ id: 'a', suite: 'Triton' }] };
  const before = structuredClone(input);
  const ordered = [original, retry, empty, newer, nullable, omitted];
  const scope = { targets: ['gfx1250'], suites: ['Triton'] };
  for (const [filters, modes] of [
    [scope, ['single', 'default']], [{ ...scope, modes: null }, ['single', 'default']],
    [{ ...scope, modes: ['single'] }, ['single']], [{ ...scope, modes: ['default'] }, ['default']],
    [{ ...scope, modes: [] }, []], [{ ...scope, targets: [] }, []],
    [{ ...scope, targets: ['unpublished-target'] }, ['single', 'default']],
    [{ ...scope, suites: [] }, ['single', 'default']],
    [{ ...scope, targets: ['gfx1250', 'gfx1250'], modes: ['default', 'single', 'single'] }, ['default', 'single', 'single']],
  ]) {
    const model = selectBenchmarkSeries(input, filters, 'a');
    expect(model.runs).toEqual(ordered);
    model.runs.forEach((run, index) => expect(run).toBe(ordered[index]));
    expect(model.labels[0]).toContain('1/2');
    expect(model.labels[1]).toContain('2/2');
    expect(model.series.map(({ target, mode }) => [target, mode])).toEqual(filters.targets.flatMap((target) => modes.map((mode) => [target, mode])));
    for (const series of model.series) {
      const expected = series.mode === 'single' ? [first, timeout, null, nonfinite, null, null] : [zero, null, null, measured, null, null];
      series.records.forEach((record, index) => {
        const test = filters.suites.length && series.target === 'gfx1250' ? expected[index] : null;
        if (!test) { expect(record).toBeNull(); expect(series.points[index]).toBeNull(); return; }
        expect(record).toEqual({ run: ordered[index], test });
        expect(record.run).toBe(ordered[index]);
        expect(record.test).toBe(test);
        if (test.status === 'completed' && Number.isFinite(test.durationSeconds)) {
          expect(series.points[index]).toEqual({ value: test.durationSeconds, record });
          expect(series.points[index].record).toBe(record);
        } else expect(series.points[index]).toBeNull();
      });
    }
  }
  const missing = selectBenchmarkSeries(input, scope, 'missing-workload');
  expect(missing.series.every(({ records, points }) => records.every((record) => record === null) && points.every((point) => point === null))).toBe(true);
  expect(input).toEqual(before);
});

test('recent history includes only the latest 20 canonical executions and preserves failure denominators', () => {
  const recent = selectRecentRunSummaries(selectRecentRunAttempts(data).slice(0, 20), filters);
  expect(recent).toHaveLength(20);
  expect(recent[0].run.runId).toBe('fictional-develop-21');
  expect(recent.every(({ run }) => run.branch === 'develop')).toBe(true);
  const failure = selectRecentRunSummaries(selectRecentRunAttempts(data), filters).find(({ run }) => run.runId === 'fictional-develop-08');
  expect(failure.completed).toBeLessThan(failure.total);
  expect(failure.duration).toBeNull();
});

test('zero baseline is measured and matched but its percentage is unavailable', () => {
  const scope = { targets: ['gfx1250'], modes: ['single'], suites: ['Triton'] };
  const baseline = structuredClone(run('14'));
  baseline.tests.find((test) => test.target === 'gfx1250' && test.mode === 'single' && test.logicalTestId === 'a').durationSeconds = 0;
  const result = selectRunComparison(run('15'), baseline, scope);
  const zero = result.comparable.find(({ baselineTest }) => baselineTest.logicalTestId === 'a');
  expect(zero.delta).toBeNull();
  expect(zero.deltaSeconds).toBe(run('15').tests[0].durationSeconds);
  expect(result.notComparable).toHaveLength(0);
  expect(result.counts.unavailable).toBe(1);
});
