import { fork } from 'node:child_process';
import { expect, test, vi } from 'vitest';
import { selectOverviewProjection } from '../../src/data/overviewProjection.js';

function boundedEmptyCalendar(generatedAt, range, scenario = 'empty') {
  return new Promise((resolve) => {
    const child = fork(new URL('../helpers/overview-calendar-child.js', import.meta.url), [generatedAt, range, scenario], { silent: true, execArgv: [] });
    const events = [];
    let stderr = ''; let spawnError = null; let timeoutStage = null;
    const started = Date.now();
    let timer = setTimeout(() => { timeoutStage = 'setup'; child.kill('SIGKILL'); }, 5_000);
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.once('error', (error) => { spawnError = error.message; });
    child.on('message', (message) => {
      events.push(message);
      if (message.stage === 'validated-before-projection') {
        clearTimeout(timer);
        timer = setTimeout(() => { timeoutStage = 'projection'; child.kill('SIGKILL'); }, 500);
      }
    });
    child.once('close', (code, signal) => {
      clearTimeout(timer);
      let reaped = false;
      try { process.kill(child.pid, 0); } catch (error) { reaped = error.code === 'ESRCH'; }
      const evidence = { pid: child.pid, generatedAt, range, scenario, setupLimitMs: 5_000, projectionLimitMs: 500,
        elapsedMs: Date.now() - started, events, code, signal, timeoutStage, stderr, spawnError, closed: true, reaped };
      console.info('bounded-calendar-child', JSON.stringify(evidence));
      resolve(evidence);
    });
  });
}

const calendarCases = [
  { label: 'ordinary validated control', generatedAt: '2026-01-07T00:00:00Z', range: '1W', days: ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05', '2026-01-06', '2026-01-07'] },
  { label: 'upper-year weekly', generatedAt: '9999-12-31T00:00:00Z', range: '1W', days: ['9999-12-25', '9999-12-26', '9999-12-27', '9999-12-28', '9999-12-29', '9999-12-30', '9999-12-31'] },
  { label: 'upper-year all', generatedAt: '9999-12-31T00:00:00Z', range: 'ALL', days: ['9999-12-31'] },
  { label: 'lower-year weekly', generatedAt: '0000-01-01T00:00:00Z', range: '1W', days: [
    '-000001-12-26', '-000001-12-27', '-000001-12-28', '-000001-12-29', '-000001-12-30', '-000001-12-31', '0000-01-01',
  ] },
  { label: 'lower-year monthly', generatedAt: '0000-01-01T00:00:00Z', range: '1M', days: [
    '-000001-12-03', '-000001-12-04', '-000001-12-05', '-000001-12-06', '-000001-12-07', '-000001-12-08', '-000001-12-09',
    '-000001-12-10', '-000001-12-11', '-000001-12-12', '-000001-12-13', '-000001-12-14', '-000001-12-15', '-000001-12-16',
    '-000001-12-17', '-000001-12-18', '-000001-12-19', '-000001-12-20', '-000001-12-21', '-000001-12-22', '-000001-12-23',
    '-000001-12-24', '-000001-12-25', '-000001-12-26', '-000001-12-27', '-000001-12-28', '-000001-12-29', '-000001-12-30',
    '-000001-12-31', '0000-01-01',
  ] },
];
test.each(calendarCases)('bounded empty calendar: $label terminates with exact inclusive keys', async ({ generatedAt, range, days }) => {
  const evidence = await boundedEmptyCalendar(generatedAt, range);
  expect(evidence.events[0]).toEqual({ stage: 'validated-before-projection', generatedAt, runs: 0, range });
  expect(evidence.spawnError).toBeNull();
  expect(evidence.stderr).toBe('');
  expect(evidence.closed).toBe(true);
  expect(evidence.reaped).toBe(true);
  expect(evidence.timeoutStage, 'validated real projection must finish before the independently enforced child deadline').toBeNull();
  expect([evidence.code, evidence.signal]).toEqual([0, null]);
  expect(evidence.events[1]).toEqual({ stage: 'result', anchorDay: days.at(-1), startDay: days[0], dayKeys: days });
  expect(evidence.events[1].dayKeys).toHaveLength(range === '1M' ? 30 : range === '1W' ? 7 : 1);
  expect(new Set(evidence.events[1].dayKeys).size).toBe(days.length);
}, 8_000);

test.each([
  { anchor: '2026-01-03', days: ['2025-12-28', '2025-12-29', '2025-12-30', '2025-12-31', '2026-01-01', '2026-01-02', '2026-01-03'] },
  { anchor: '2024-03-03', days: ['2024-02-26', '2024-02-27', '2024-02-28', '2024-02-29', '2024-03-01', '2024-03-02', '2024-03-03'] },
  { anchor: '2025-03-03', days: ['2025-02-25', '2025-02-26', '2025-02-27', '2025-02-28', '2025-03-01', '2025-03-02', '2025-03-03'] },
])('inclusive calendar preserves ordinary month/year/leap boundary ending $anchor', ({ anchor, days }) => {
  expect(selectOverviewProjection({ runs: [], generatedAt: `${anchor}T00:00:00Z` }, filters, { range: '1W' }).dayKeys).toEqual(days);
});

// Independent reference uses numeric UTC-midnight ticks, never projection helpers.
function referenceUtcDays(start, end) {
  const first = Date.parse(`${start}T00:00:00Z`);
  const last = Date.parse(`${end}T00:00:00Z`);
  return Array.from({ length: (last - first) / 86_400_000 + 1 }, (_, index) =>
    new Date(first + index * 86_400_000).toISOString().split('T')[0]);
}

const offsetBoundaries = [
  { label: 'upper', generatedAt: '9999-12-31T23:30:00-01:00', anchor: '+010000-01-01', yearStart: '+010000-01-01',
    populatedAt: '9999-12-31T22:45:00-02:00', earliest: '9999-12-31',
    timestamps: ['9999-12-31T12:00:00Z', '9999-12-31T23:30:00-01:00', '9999-12-31T22:45:00-02:00'] },
  { label: 'lower', generatedAt: '0000-01-01T00:30:00+01:00', anchor: '-000001-12-31', yearStart: '-000001-01-01',
    populatedAt: '0000-01-01T01:15:00+01:00', populatedAnchor: '0000-01-01', populatedYearStart: '0000-01-01', earliest: '-000001-12-31',
    timestamps: ['0000-01-01T00:30:00+01:00', '0000-01-01T00:00:00Z', '0000-01-01T01:15:00+01:00'] },
];
const offsetCases = offsetBoundaries.flatMap((boundary) => ['empty', 'populated'].flatMap((kind) =>
  ['ALL', '1W', '1M', '3M', '1D', '6M', 'YTD'].map((range) => ({ ...boundary, kind, range }))));

test.each(offsetCases)('bounded offset calendar: $label $kind $range preserves UTC identities and chronological membership', async (input) => {
  const populated = input.kind === 'populated';
  const generatedAt = populated ? input.populatedAt : input.generatedAt;
  const anchor = populated ? input.populatedAnchor ?? input.anchor : input.anchor;
  const rangeDays = { '1W': 7, '1M': 30, '3M': 90, '1D': 1, '6M': 180 };
  const start = input.range === 'ALL' ? populated ? input.earliest : anchor
    : input.range === 'YTD' ? populated ? input.populatedYearStart ?? input.yearStart : input.yearStart
      : new Date(Date.parse(`${anchor}T00:00:00Z`) - (rangeDays[input.range] - 1) * 86_400_000).toISOString().split('T')[0];
  const days = referenceUtcDays(start, anchor);
  const evidence = await boundedEmptyCalendar(generatedAt, input.range, populated ? input.label : 'empty');
  expect(evidence.events[0]).toEqual({ stage: 'validated-before-projection', generatedAt, runs: populated ? 3 : 0, range: input.range });
  expect(evidence.spawnError).toBeNull();
  expect(evidence.stderr).toBe('');
  expect(evidence.closed).toBe(true);
  expect(evidence.reaped).toBe(true);
  expect(evidence.timeoutStage).toBeNull();
  expect([evidence.code, evidence.signal]).toEqual([0, null]);
  const result = evidence.events[1];
  expect({ anchorDay: result.anchorDay, startDay: result.startDay, dayKeys: result.dayKeys })
    .toEqual({ anchorDay: anchor, startDay: start, dayKeys: days });
  expect(result.dayKeys).toHaveLength(days.length);
  expect(new Set(result.dayKeys).size).toBe(days.length);
  if (input.label === 'upper' && input.range === '1W') {
    expect(result.dayKeys).toEqual(['9999-12-26', '9999-12-27', '9999-12-28', '9999-12-29', '9999-12-30', '9999-12-31', '+010000-01-01']);
  }
  if (populated) {
    const indexes = input.range === '1W' ? [0, 1, 2] : input.range === '1D' ? [1, 2] : input.range === 'YTD' ? [2] : [0, 2];
    const identities = indexes.map((index) => ({ id: `synthetic-run-000${index}`, dayKey: index === 0 ? input.earliest : anchor }));
    expect(result.populated.attempts).toEqual(identities.map((identity) => ({ ...identity, eligible: true })));
    expect(result.populated.slots).toEqual(identities);
    expect(result.populated.cohort).toEqual(['gemm-bf16-4096', 'gemm-f16-1024']);
    expect(result.populated.points).toEqual(identities.map((identity, position) => ({ ...identity, durationSeconds: [3, 3.002, 3.004][indexes[position]] })));
    expect(result.populated.timestamps).toEqual(input.timestamps);
    expect(result.populated.originalInputsUnchanged).toBe(true);
    expect(result.populated.sourceLeavesRetained).toBe(true);
    expect(result.populated.independentWrappers).toBe(true);
  }
}, 8_000);

const filters = { targets: ['gfx950', 'gfx1250'], modes: ['single'], suites: ['gemm'] };
const catalog = (id, configurations) => ({ id, configurations, tests: ['a', 'b', 'c'].map((id) => ({ id, suite: id === 'c' ? 'fft' : 'gemm' })) });
function run(id, day, catalog, entries, { sha = id, completion = `${day}T12:00:00Z`, branch = 'develop' } = {}) {
  return { runId: id, branch, catalogId: catalog.id,
    commitTimestamp: `${day}T10:00:00Z`, timestamp: completion, provenance: { rocjitsuCommitSha: sha },
    configurations: Object.keys(entries).map((key) => { const [target, thread_policy] = key.split('/'); return { target, thread_policy }; }),
    tests: Object.entries(entries).flatMap(([key, durations]) => Object.entries(durations).map(([id, value]) => {
      const [target, mode] = key.split('/');
      return { testId: `${key}:${id}`, logicalTestId: id, target, mode, suite: id === 'c' ? 'fft' : 'gemm',
        status: value === null ? 'failed' : 'completed', durationSeconds: value };
    })) };
}
const data = (runs, catalogs) => ({ runs, catalogs: Object.fromEntries(catalogs.map((c) => [c.id, c])) });

test('daily cohorts intersect chosen catalogs per configuration before result eligibility; latest failure stays a gap', () => {
  const a = catalog('A', { 'gfx950/single': ['a', 'b'], 'gfx1250/single': ['b'] });
  const b = catalog('B', { 'gfx950/single': ['a', 'c'] });
  const c = catalog('C', { 'gfx950/single': ['a', 'b'], 'gfx1250/single': ['b'] });
  const runs = [run('one', '2026-01-01', a, { 'gfx950/single': { a: 4, b: 100 }, 'gfx1250/single': { b: 2 } }),
    run('two', '2026-01-02', b, { 'gfx950/single': { a: 2, c: 99 } }),
    run('three', '2026-01-03', c, { 'gfx950/single': { a: null, b: 3 }, 'gfx1250/single': { b: 1 } })];
  const model = selectOverviewProjection(data(runs, [a, b, c]), filters, { range: 'ALL' });
  expect(model.rows[0].cohortIds).toEqual(['b']); // Target ordering, not publication order.
  const row = model.rows.find((r) => r.target === 'gfx950');
  expect(row.cohortIds).toEqual(['a']);
  expect(row.points.map((p) => p.durationSeconds)).toEqual([4, 2, null]);
  expect(row.points.map((p) => p.speedup)).toEqual([1, 2, null]);
  expect(row.latest.completed).toBe(0);
  expect(row.latest.required).toBe(1);
  expect(row.latest.reason).toBe('failed');
  expect(model.rows[0].points[1].reason).toBe('configuration-absent');
  expect(model.rows[0].latest.speedup).toBe(2);
  expect(selectOverviewProjection(data(runs, [a, b, c]), { ...filters, targets: ['gfx950'] }).rows[0].cohortIds).toEqual(['a']);
});

test('daily picks one latest commit/attempt before configuration projection, including UTC boundaries and backfills', () => {
  const c = catalog('C', { 'gfx950/single': ['a'] });
  const older = run('older', '2026-01-01', c, { 'gfx950/single': { a: 1 } }, { completion: '2026-01-04T12:00:00Z' });
  const first = run('first', '2026-01-01', c, { 'gfx950/single': { a: 2 } });
  first.commitTimestamp = '2026-01-01T23:59:59Z';
  const failed = { ...first, runId: 'retry', timestamp: '2026-01-02T01:00:00Z', tests: first.tests.map((t) => ({ ...t, status: 'timeout', durationSeconds: null })) };
  const boundary = run('boundary', '2026-01-02', c, { 'gfx950/single': { a: 4 } });
  boundary.commitTimestamp = '2026-01-02T00:00:00Z';
  const side = run('side', '2026-02-01', c, { 'gfx950/single': { a: 99 } }, { branch: 'feature' });
  const model = selectOverviewProjection(data([side, boundary, first, older, failed], [c]), { ...filters, targets: ['gfx950'] });
  expect(model.slots.map((s) => s.run.runId)).toEqual(['retry', 'boundary']);
  expect(model.rows[0].points[0].reason).toBe('timeout');
  expect(model.rows[0].reference.runId).toBe('boundary');
  expect(model.anchorDay).toBe('2026-01-02');
});

test('equal commit timestamps choose SHA before completion and latest attempt within that commit', () => {
  const c = catalog('C', { 'gfx950/single': ['a'] });
  const runs = [
    run('b-first', '2026-01-01', c, { 'gfx950/single': { a: 2 } }, { sha: 'bbbb', completion: '2026-01-01T11:00:00Z' }),
    run('a-later', '2026-01-01', c, { 'gfx950/single': { a: 4 } }, { sha: 'aaaa', completion: '2026-01-01T12:00:00Z' }),
  ];
  const project = () => selectOverviewProjection(data(runs, [c]), { ...filters, targets: ['gfx950'] });
  expect(project().latestRun.runId).toBe('b-first');
  expect(project().rows[0].latest.durationSeconds).toBe(2);
  runs.push(run('b-retry', '2026-01-01', c, { 'gfx950/single': { a: 1 } }, { sha: 'bbbb', completion: '2026-01-01T11:30:00Z' }));
  expect(project().latestRun.runId).toBe('b-retry');
  expect(project().rows[0].latest.durationSeconds).toBe(1);
});

test('catalog switching preserves the full-history date window and deterministic daily tie-breaks', () => {
  const a = catalog('A', { 'gfx950/single': ['a'] });
  const b = catalog('B', { 'gfx950/single': ['a'] });
  const runs = [run('old', '2026-01-01', a, { 'gfx950/single': { a: 8 } }),
    run('z', '2026-01-20', b, { 'gfx950/single': { a: 2 } }, { sha: 'z' }),
    run('a', '2026-01-20', b, { 'gfx950/single': { a: 4 } }, { sha: 'a' }),
    run('zz', '2026-01-20', b, { 'gfx950/single': { a: 1 } }, { sha: 'z' })];
  const all = selectOverviewProjection(data(runs, [a, b]), filters, { range: '1M' });
  const scoped = selectOverviewProjection(data(runs, [a, b]), filters, { range: '1M', catalogId: 'A' });
  expect(scoped.dayKeys).toEqual(all.dayKeys);
  expect(all.latestRun.runId).toBe('zz');
  expect(scoped.latestRun.runId).toBe('old');
  expect(scoped.rows.find((r) => r.target === 'gfx950').latest.durationSeconds).toBe(8);
});

test('empty selections/intersections and invalid totals never become zero or a success fallback', () => {
  const a = catalog('A', { 'gfx950/single': ['a'] });
  const b = catalog('B', { 'gfx950/single': ['b'] });
  const runs = [run('one', '2026-01-01', a, { 'gfx950/single': { a: 2 } }),
    run('two', '2026-01-02', b, { 'gfx950/single': { b: 1 } })];
  const source = data(runs, [a, b]);
  expect(selectOverviewProjection(source, { ...filters, targets: [] }).rows).toEqual([]);
  expect(selectOverviewProjection(source, { ...filters, modes: [] }).rows).toEqual([]);
  const empty = selectOverviewProjection(source, filters).rows.find((r) => r.target === 'gfx950');
  expect(empty.cohortIds).toEqual([]);
  expect(empty.latest.durationSeconds).toBeNull();
  expect(empty.latest.reason).toBe('no-comparable-workloads');
  expect(selectOverviewProjection(source, { ...filters, suites: [] }).rows.every((r) => r.reference === null)).toBe(true);
  const c = catalog('C', { 'gfx950/single': ['a', 'b'] });
  const invalid = run('invalid', '2026-01-03', c, { 'gfx950/single': { a: 0, b: 2 } });
  expect(selectOverviewProjection(data([invalid], [c]), filters).rows.find((r) => r.target === 'gfx950').reference).toBeNull();
  invalid.tests.forEach((t) => { t.durationSeconds = 1e308; });
  const tiny = run('tiny', '2026-01-01', c, { 'gfx950/single': { a: 5e-324, b: 5e-324 } });
  const large = run('large', '2026-01-02', c, { 'gfx950/single': { a: 4e15, b: 4e15 } });
  expect(selectOverviewProjection(data([tiny, large], [c]), filters).rows.find((r) => r.target === 'gfx950').durationChangePercent).toBeNull();
  const overflow = selectOverviewProjection(data([invalid], [c]), filters).rows.find((r) => r.target === 'gfx950');
  expect(overflow.latest.reason).toBe('invalid-total');
  expect(overflow.latest.durationSeconds).toBeNull();
});

test('row order is policies first then deterministic targets and missing results retain required counts', () => {
  const c = catalog('C', { 'gfx950/single': ['a', 'b'], 'gfx950/default': ['a'] });
  const r = run('one', '2026-01-01', c, { 'gfx950/single': { a: 1 }, 'gfx950/default': { a: 2 } });
  const model = selectOverviewProjection(data([r], [c]), { ...filters, targets: ['gfx950', 'gfx1250'], modes: ['default', 'single'] });
  expect(model.rows.map((r) => r.key)).toEqual(['gfx1250/single', 'gfx950/single', 'gfx1250/default', 'gfx950/default']);
  expect(model.rows[1].latest).toMatchObject({ required: 2, completed: 1, reason: 'missing-result', durationSeconds: null });
  expect(model.rows[3].latest.durationSeconds).toBe(2);
});

test('weekly retains repeated SHA attempts without cap and calculates cohort before removing incomplete attempts', () => {
  const a = catalog('A', { 'gfx950/single': ['a', 'b'] });
  const b = catalog('B', { 'gfx950/single': ['a'] });
  const runs = Array.from({ length: 10 }, (_, i) => run(`attempt-${i}`, '2026-01-07', a,
    { 'gfx950/single': { a: 2, b: 5 } }, { sha: 'same', completion: `2026-01-07T12:00:${String(i).padStart(2, '0')}Z` }));
  runs.push(run('failed', '2026-01-07', b, { 'gfx950/single': { a: null } }));
  const model = selectOverviewProjection(data(runs, [a, b]), { ...filters, targets: ['gfx950'] }, { range: '1W' });
  expect(model.attempts).toHaveLength(11);
  expect(model.attempts.find((a) => a.runId === 'failed').eligible).toBe(false);
  expect(model.rows[0].cohortIds).toEqual(['a']);
  expect(model.slots.filter((s) => s.run)).toHaveLength(10);
  expect(model.rows[0].latest.runId).toBe('attempt-9');
  expect(model.endpointLabel).toBe('Latest eligible plotted run');
  expect(model.weeklyEligibility).toBe('selected-workloads-succeed');
});

const oneLane = { targets: ['gfx950'], modes: ['single'], suites: ['gemm'] };

test.each(['1W', '1M', '3M', 'ALL', '1D', '6M', 'YTD', 'unknown', null, undefined])(
  'projection values and endpoint aliases survive range %s', (range) => {
    const c = catalog('C', { 'gfx950/single': ['a'] });
    const first = run('first', '2026-01-06', c, { 'gfx950/single': { a: 8 } });
    const second = run('second', '2026-01-07', c, { 'gfx950/single': { a: 4 } });
    const source = data([first, second], [c]);
    const before = JSON.stringify(source);
    const model = selectOverviewProjection(source, oneLane, { range });
    const selected = range === '1D' ? [second] : [first, second];
    expect(model.range).toBe(['unknown', null, undefined].includes(range) ? 'ALL' : range);
    expect(model.anchorDay).toBe('2026-01-07');
    expect(model.slots.filter((s) => s.run).map((s) => s.run)).toEqual(selected);
    const row = model.rows[0];
    const measured = row.points.filter((p) => p.run);
    expect(measured.map((p) => [p.runId, p.durationSeconds, p.speedup, p.required, p.completed, p.reason]))
      .toEqual(range === '1D' ? [['second', 4, 1, 1, 1, null]]
        : [['first', 8, 1, 1, 1, null], ['second', 4, 2, 1, 1, null]]);
    expect(row.durationChangePercent).toBe(range === '1D' ? 0 : -50);
    expect(row.reference).toBe(measured[0]);
    expect(row.latest).toBe(measured.at(-1));
    expect(row.endpoints.reference).toBe(row.reference);
    expect(row.endpoints.latest).toBe(row.latest);
    for (const [i, point] of measured.entries()) {
      expect(point.run).toBe(selected[i]);
      expect(point.results[0].result).toBe(selected[i].tests[0]);
    }
    expect(model.attempts.every((a) => a.points.every((p) => p.speedup === null))).toBe(true);
    expect(JSON.stringify(source)).toBe(before);
  },
);

test.each([null, 'A', 'absent'])('catalog %s preserves window, gaps and no-run fallback ownership', (catalogId) => {
  const a = catalog('A', { 'gfx950/single': ['a'] });
  const b = catalog('B', { 'gfx950/single': ['a'] });
  const first = run('first', '2026-01-06', a, { 'gfx950/single': { a: 8 } });
  const second = run('second', '2026-01-07', b, { 'gfx950/single': { a: 4 } });
  const model = selectOverviewProjection(data([first, second], [a, b]), oneLane, { range: 'ALL', catalogId });
  expect([model.startDay, model.anchorDay, model.dayKeys]).toEqual(['2026-01-06', '2026-01-07', ['2026-01-06', '2026-01-07']]);
  expect(model.attempts.map((a) => a.runId)).toEqual(catalogId === 'absent' ? [] : catalogId === 'A' ? ['first'] : ['first', 'second']);
  const row = model.rows[0];
  if (catalogId === 'absent') {
    expect(row.reference).toBeNull();
    expect(row.latest).not.toBe(row.points[0]);
    expect(row.latest).not.toBe(row.points[1]);
    expect(row.latest).toMatchObject({ run: null, dayKey: null, speedup: null, reason: 'no-run', results: [], reasons: [] });
    expect(row.points.map((p) => [p.dayKey, p.reason])).toEqual([['2026-01-06', 'no-run'], ['2026-01-07', 'no-run']]);
    expect(row.points[0].results).not.toBe(row.points[1].results);
    expect(row.latest.results).not.toBe(row.points[0].results);
    expect(row.endpoints.latest).toBe(row.latest);
  } else {
    expect(row.latest).toBe(row.points[catalogId === 'A' ? 0 : 1]);
    expect(row.latest.durationSeconds).toBe(catalogId === 'A' ? 8 : 4);
  }
});

test('attempt and plotted wrappers isolate mutation in both directions while preserving source leaves', () => {
  const c = catalog('C', { 'gfx950/single': ['a', 'b'] });
  const first = run('first', '2026-01-06', c, { 'gfx950/single': { a: 8, b: 2 } });
  const failed = run('failed', '2026-01-07', c, { 'gfx950/single': { a: null, b: 1 } });
  const source = data([first, failed], [c]);
  const before = JSON.stringify(source);
  const model = selectOverviewProjection(source, oneLane, { range: 'ALL' });
  const row = model.rows[0];
  for (let i = 0; i < model.attempts.length; i += 1) {
    const attempt = model.attempts[i].points[0];
    const plotted = row.points[i];
    expect(attempt).not.toBe(plotted);
    expect(attempt.results).not.toBe(plotted.results);
    expect(attempt.reasons).not.toBe(plotted.reasons);
    for (let j = 0; j < attempt.results.length; j += 1) {
      expect(attempt.results[j]).not.toBe(plotted.results[j]);
      expect(attempt.results[j].result).toBe(source.runs[i].tests[j]);
      expect(plotted.results[j].result).toBe(source.runs[i].tests[j]);
    }
    plotted.results[0].id = 'row-only';
    expect(attempt.results[0].id).toBe('a');
    attempt.results[1].reason = 'attempt-only';
    expect(plotted.results[1].reason).toBeNull();
    plotted.results.push({ id: 'row-extra' });
    expect(attempt.results).toHaveLength(2);
    attempt.reasons.push({ id: 'attempt-extra' });
    expect(plotted.reasons).toHaveLength(i === 0 ? 0 : 1);
    expect(attempt.speedup).toBeNull();
  }
  const failure = model.attempts[1].points[0];
  expect(failure.reasons[0]).not.toBe(row.points[1].reasons[0]);
  row.points[1].reasons[0].reason = 'row-reason';
  expect(failure.reasons[0].reason).toBe('failed');
  failure.reasons[0].id = 'attempt-reason';
  expect(row.points[1].reasons[0].id).toBe('a');
  expect(JSON.stringify(source)).toBe(before);
  const again = selectOverviewProjection(source, oneLane, { range: 'ALL' });
  expect(again.rows[0].points[1]).toMatchObject({ reason: 'failed', results: [{ id: 'a' }, { id: 'b' }] });
  expect(again.attempts[1].points[0].results[0].result).toBe(failed.tests[0]);
});

test('repeated object occurrences retain independent attempt, slot and plotted containers', () => {
  const c = catalog('C', { 'gfx950/single': ['a'] });
  const r = run('same', '2026-01-07', c, { 'gfx950/single': { a: 2 } });
  const model = selectOverviewProjection(data([r, r], [c]), oneLane, { range: '1W' });
  expect(model.attempts).toHaveLength(2);
  const slots = model.slots.filter((s) => s.run);
  expect(slots).toHaveLength(2);
  expect(slots[0]).not.toBe(slots[1]);
  expect(slots.map((s) => s.x)).toEqual([6 + 1 / 3, 6 + 2 / 3]);
  const attempts = model.attempts.map((a) => a.points[0]);
  expect(model.attempts[0]).not.toBe(model.attempts[1]);
  expect(model.attempts[0].points).not.toBe(model.attempts[1].points);
  const points = [...attempts, ...model.rows[0].points.filter((p) => p.run)];
  expect(new Set(points).size).toBe(4);
  expect(new Set(points.map((p) => p.results)).size).toBe(4);
  expect(new Set(points.map((p) => p.reasons)).size).toBe(4);
  expect(new Set(points.map((p) => p.results[0])).size).toBe(4);
  expect(points.every((p) => p.run === r && p.results[0].result === r.tests[0])).toBe(true);
  points[0].results[0].id = 'first-only';
  expect(points.slice(1).every((p) => p.results[0].id === 'a')).toBe(true);
  expect(attempts.every((p) => p.speedup === null)).toBe(true);
});

test('distinct run objects with matching runId and SHA do not collapse weekly measurements', () => {
  const c = catalog('C', { 'gfx950/single': ['a'] });
  const first = run('same', '2026-01-07', c, { 'gfx950/single': { a: 8 } });
  const second = run('same', '2026-01-07', c, { 'gfx950/single': { a: 4 } }, { completion: '2026-01-07T13:00:00Z' });
  const row = selectOverviewProjection(data([second, first], [c]), oneLane, { range: '1W' }).rows[0];
  const points = row.points.filter((p) => p.run);
  expect(points.map((p) => [p.durationSeconds, p.speedup])).toEqual([[8, 1], [4, 2]]);
  expect(points[0].run).toBe(first);
  expect(points[1].run).toBe(second);
});

test.each(['failed', 'timeout', 'missing-result', 'configuration-absent', 'invalid-duration', 'invalid-total'])(
  'attempt/row values retain %s failure semantics', (reason) => {
    const c = catalog('C', { 'gfx950/single': ['a', 'b'] });
    const r = run('one', '2026-01-07', c, { 'gfx950/single': { a: 2, b: 3 } });
    if (reason === 'failed' || reason === 'timeout') r.tests[0].status = reason;
    if (reason === 'missing-result') r.tests.shift();
    if (reason === 'configuration-absent') r.configurations = [];
    if (reason === 'invalid-duration') r.tests[0].durationSeconds = NaN;
    if (reason === 'invalid-total') r.tests.forEach((t) => { t.durationSeconds = 1e308; });
    const model = selectOverviewProjection(data([r], [c]), oneLane, { range: 'ALL' });
    const actualReason = reason === 'configuration-absent' ? 'no-comparable-workloads' : reason;
    expect(model.attempts[0].eligible).toBe(false);
    expect(model.attempts[0].points[0]).toMatchObject({ reason: actualReason, speedup: null, durationSeconds: null });
    expect(model.rows[0].latest).toMatchObject({ reason: actualReason, speedup: null, durationSeconds: null });
    expect(model.rows[0].reference).toBeNull();
  },
);

test('empty and generatedAt-only models preserve calendar gaps, fallback and empty filter semantics', () => {
  const c = catalog('C', { 'gfx950/single': ['a'] });
  const source = { ...data([], [c]), generatedAt: '2026-01-07T12:00:00Z' };
  const generated = selectOverviewProjection(source, oneLane, { range: '1W' });
  expect(generated.dayKeys).toEqual(['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05', '2026-01-06', '2026-01-07']);
  expect(generated.rows[0].points.map((p) => p.dayKey)).toEqual(generated.dayKeys);
  expect(generated.rows[0].latest.dayKey).toBeNull();
  expect(generated.rows[0].points).not.toContain(generated.rows[0].latest);
  expect(selectOverviewProjection(data([], []), oneLane).rows[0]).toMatchObject({ points: [], reference: null, latest: { reason: 'no-run', dayKey: null } });
  const r = run('one', '2026-01-07', c, { 'gfx950/single': { a: 2 } });
  const populated = data([r], [c]);
  expect(selectOverviewProjection(populated, { ...oneLane, targets: [] }).rows).toEqual([]);
  expect(selectOverviewProjection(populated, { ...oneLane, modes: [] }).attempts[0].points).toEqual([]);
  expect(selectOverviewProjection(populated, { ...oneLane, suites: [] }).rows[0].latest.reason).toBe('no-comparable-workloads');
  const { modes: _modes, ...withoutModes } = oneLane;
  const omitted = selectOverviewProjection(populated, withoutModes);
  expect(omitted.rows.map((r) => r.key)).toEqual(['gfx950/single', 'gfx950/default']);
  expect(selectOverviewProjection(populated, { ...oneLane, modes: null })).toEqual(omitted);
});

test.each(['ALL', '1W'])('heavy scan runs only once per candidate/configuration for %s', (range) => {
  const c = catalog('C', { 'gfx950/single': ['a'] });
  const r = run('one', '2026-01-07', c, { 'gfx950/single': { a: 2 } });
  const scan = vi.spyOn(r.tests, 'filter');
  try {
    const model = selectOverviewProjection(data([r], [c]), oneLane, { range });
    expect(model.attempts[0].eligible).toBe(true);
    expect(model.rows[0].latest.durationSeconds).toBe(2);
    expect(scan).toHaveBeenCalledTimes(1);
  } finally {
    scan.mockRestore();
  }
});
