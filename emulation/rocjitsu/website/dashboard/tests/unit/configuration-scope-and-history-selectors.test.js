import { fixtureRunPath } from '../fixtures/runPath.js';
import { expect, test } from 'vitest';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { selectOverview, selectRecentRunAttempts, selectRecentRunSummaries, selectBenchmarkSeries, selectBenchmarkCatalog, selectRunComparison } from '../../src/data/selectors.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

const data = validatePublishedDashboardData(createSchema2Publication()).data;
const filters = { targets: ['gfx1250', 'gfx950'], suites: data.suites, modes: ['ST', 'MT'] };
const run = (id) => data.runs.find(({ runId }) => runId === `fictional-develop-${id}`);

test('Overview uses one atomic selected-scope sum with first-success current-catalog anchors', () => {
  const overview = selectOverview(data, filters, 'ALL');
  const { history, metrics } = overview;
  expect(history.series).toHaveLength(1);
  expect(history.series[0].target).toBe('Selected runtime');
  const first = history.slots.findIndex(({ run }) => run?.runId === 'fictional-develop-00');
  const expected = filters.targets.flatMap((target) => filters.modes.map((mode) => {
    const measured = run('00').tests.filter((t) => t.target === target && t.mode === mode && ['a', 'b'].includes(t.logicalTestId));
    const newIds = target === 'gfx950' && mode === 'MT' ? ['d'] : ['d', 'e'];
    return measured.reduce((s, t) => s + t.durationSeconds, 0) + newIds.reduce((sum, id) => {
      const anchor = [...data.runs].sort((a, b) => a.timestamp.localeCompare(b.timestamp)).find((r) => r.catalogId === 'fictional-current' && r.tests.some((t) => t.target === target && t.mode === mode && t.logicalTestId === id && t.status === 'completed'));
      return sum + anchor.tests.find((t) => t.target === target && t.mode === mode && t.logicalTestId === id).durationSeconds;
    }, 0);
  })).reduce((sum, value) => sum + value, 0);
  expect(history.series[0].data[first]).toBeCloseTo(expected);
  expect(history.series[0].baseline).toBeCloseTo(expected);
  expect(history.anchors.find((a) => a.testId === 'gfx1250:ST:d').runId).toBe('fictional-develop-09');
  expect(history.anchors.find((a) => a.testId === 'gfx1250:ST:e').runId).toBe('fictional-develop-10');
  expect(history.anchors.find((a) => a.testId === 'gfx1250:MT:d').runId).toBe('fictional-develop-08');
  expect(history.series[0].data[history.slots.findIndex(({ run }) => run?.runId === 'fictional-develop-08')]).toBeNull();
  expect(history.series[0].data[history.slots.findIndex(({ run }) => run?.runId === 'fictional-develop-12')]).toBeNull();
  expect(history.currentDuration).toBe(history.series[0].data.at(-1));
  expect(metrics.duration).toBe(history.currentDuration);
  expect(metrics.durationDelta).toBe(history.durationDelta);
  expect(overview.metricsBaseline).toBe(history.firstRun);
});

test('timeframe cards, endpoint captions and baseline follow the represented period', () => {
  const { history, metrics, metricsBaseline } = selectOverview(data, filters, '1W');
  const real = history.series[0].data.filter(Number.isFinite);
  expect(history.series[0].baseline).toBe(real[0]);
  expect(metrics.duration).toBe(real.at(-1));
  expect(metricsBaseline).toBe(history.firstRun);
  expect(metrics.durationDelta).toBe(history.durationDelta);
  expect(history.firstRun.commitTimestamp.slice(0, 10) >= '2026-09-28').toBe(true);
});

test('mode-specific benchmark histories keep measured zero and missing configurations as gaps', () => {
  const series = selectBenchmarkSeries(data, filters, 'a').series;
  expect(series.map(({ key, target, mode }) => ({ key, target, mode }))).toEqual([
    { key: 'gfx1250:ST', target: 'gfx1250', mode: 'ST' }, { key: 'gfx1250:MT', target: 'gfx1250', mode: 'MT' },
    { key: 'gfx950:ST', target: 'gfx950', mode: 'ST' }, { key: 'gfx950:MT', target: 'gfx950', mode: 'MT' },
  ]);
  expect(series[3].points[12]).toBeNull();
  expect(series[0].points[14].value).toBe(0);
  expect(selectBenchmarkSeries(data, { ...filters, modes: [] }, 'a').series).toEqual([]);
  expect(selectBenchmarkSeries(data, { ...filters, suites: [] }, 'a').series.every(({ points }) => points.every((p) => p === null))).toBe(true);
  expect(selectBenchmarkCatalog(data, { ...filters, modes: [] }).available).toEqual([]);
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
  const scope = { targets: ['gfx1250'], modes: ['ST'], suites: ['Triton'] };
  const result = selectRunComparison(run('15'), run('14'), scope);
  const zero = result.comparable.find(({ baselineTest }) => baselineTest.logicalTestId === 'a');
  expect(zero.delta).toBeNull();
  expect(zero.deltaSeconds).toBe(run('15').tests[0].durationSeconds);
  expect(result.notComparable).toHaveLength(0);
  expect(result.counts.unavailable).toBe(1);
});

test('empty mode scope and empty snapshot do not manufacture totals', () => {
  expect(selectOverview(data, { ...filters, modes: [] }).metrics.duration).toBeNull();
  const empty = validatePublishedDashboardData({ ...createSchema2Publication(), runs: [], index: { generatedAt: data.generatedAt, runFiles: [] } }).data;
  expect(selectOverview(empty, filters).history.currentDuration).toBeNull();
});

test.each(['gfx1250', 'gfx950'])('DATA-03 preserves historical %s MT scope when the latest catalog is ST-only', (target) => {
  const source = createSchema2Publication();
  source.runs = source.runs.slice(0, 2);
  source.index.runFiles = source.runs.map(fixtureRunPath);
  const [earlier, latest] = source.runs;
  const key = `${target}:ST`;
  const catalog = structuredClone(source.catalogs[latest.testCatalog]);
  catalog.id = 'fictional-st-only';
  catalog.configurations = { [key]: catalog.configurations[key] };
  latest.testCatalog = `test-catalogs/${catalog.id}.json`;
  source.catalogs[latest.testCatalog] = catalog;
  latest.configurations = latest.configurations.filter((configuration) => `${configuration.target}:${configuration.threadingMode}` === key);
  const scoped = validatePublishedDashboardData(source).data;
  const scope = { targets: [target], modes: ['ST', 'MT'], suites: scoped.suites };
  const measured = earlier.configurations.filter((configuration) => configuration.target === target)
    .flatMap(({ results }) => results).reduce((total, result) => total + result.durationSeconds, 0);
  const overview = selectOverview(scoped, scope, 'ALL');
  expect(overview.history.series[0].data).toEqual([measured, null]);
  expect(overview.history.series[0].baseline).toBe(measured);
  expect(overview.history.currentDuration).toBeNull();
  expect(overview.metrics.duration).toBeNull();
  expect(overview.history.durationDelta).toBeNull();
  expect(overview.history.normalized).toBe(false);
  expect(overview.history.estimates).toEqual([]);
  expect(overview.history.anchors.filter(({ mode }) => mode === 'MT')).toHaveLength(3);
  expect(overview.history.anchors.filter(({ mode }) => mode === 'MT').every((anchor) => anchor.catalogId === 'fictional-old' && anchor.runId === earlier.id)).toBe(true);
  expect(overview.history.anchors.filter(({ mode }) => mode === 'ST').every((anchor) => anchor.catalogId === catalog.id && anchor.runId === latest.id)).toBe(true);
  const mtMeasured = earlier.configurations.find((configuration) => configuration.target === target && configuration.threadingMode === 'MT')
    .results.reduce((total, result) => total + result.durationSeconds, 0);
  expect(selectOverview(scoped, { ...scope, modes: ['MT'] }).history.series[0].data).toEqual([mtMeasured, null]);
});

test('DATA-03 resolves each selected configuration catalog by commit rather than execution order', () => {
  const source = createSchema2Publication();
  source.runs = source.runs.slice(0, 3);
  source.index.runFiles = source.runs.map(fixtureRunPath);
  const [earlier, mtRun, stRun] = source.runs;
  mtRun.execution.completedAt = '2026-09-14T06:00:00.000Z';
  const memberships = { 'gfx1250:MT': ['b', 'c'], 'gfx1250:ST': ['a', 'b'] };
  for (const [run, mode] of [[mtRun, 'MT'], [stRun, 'ST']]) {
    const key = `gfx1250:${mode}`;
    const catalog = structuredClone(source.catalogs[run.testCatalog]);
    catalog.id = `fictional-${mode.toLowerCase()}-current`;
    catalog.configurations = { [key]: memberships[key] };
    run.testCatalog = `test-catalogs/${catalog.id}.json`;
    source.catalogs[run.testCatalog] = catalog;
    run.configurations = run.configurations.filter((configuration) => `${configuration.target}:${configuration.threadingMode}` === key);
    run.configurations[0].results = run.configurations[0].results.filter(({ testId }) => memberships[key].includes(testId));
  }
  const scoped = validatePublishedDashboardData(source).data;
  expect(scoped.latestRun.runId).toBe(mtRun.id);
  expect(scoped.latestCommitRun.runId).toBe(stRun.id);
  const measured = earlier.configurations.filter(({ target }) => target === 'gfx1250').flatMap(({ threadingMode: mode, results }) =>
    results.filter(({ testId }) => memberships[`gfx1250:${mode}`].includes(testId))).reduce((total, result) => total + result.durationSeconds, 0);
  const { history } = selectOverview(scoped, { targets: ['gfx1250'], modes: ['ST', 'MT'], suites: scoped.suites });
  expect(history.series[0].data).toEqual([measured, null, null]);
  expect(history.anchors.map(({ testId, runId, catalogId }) => ({ testId, runId, catalogId }))).toEqual([
    { testId: 'gfx1250:ST:a', runId: stRun.id, catalogId: 'fictional-st-current' },
    { testId: 'gfx1250:ST:b', runId: stRun.id, catalogId: 'fictional-st-current' },
    { testId: 'gfx1250:MT:b', runId: mtRun.id, catalogId: 'fictional-mt-current' },
    { testId: 'gfx1250:MT:c', runId: mtRun.id, catalogId: 'fictional-mt-current' },
  ]);
});

test('small measured durations are not rounded into synthetic zero', () => {
  const source = createSchema2Publication();
  for (const run of source.runs) for (const configuration of run.configurations) for (const result of configuration.results) {
    if (result.status === 'completed') result.durationSeconds = 1e-9;
  }
  const tiny = validatePublishedDashboardData(source).data;
  const overview = selectOverview(tiny, { targets: ['gfx1250'], suites: tiny.suites, modes: ['ST'] });
  expect(overview.history.currentDuration).toBeCloseTo(4e-9, 15);
  expect(overview.history.currentDuration).toBeGreaterThan(0);
});
