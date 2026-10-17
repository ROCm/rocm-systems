import { expect, test } from 'vitest';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { selectOverview, selectRunComparison } from '../../src/data/selectors.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

function unanchoredData() {
  const source = createSchema2Publication();
  for (const run of source.runs) for (const config of run.configurations) {
    if (config.target !== 'gfx1250' || config.threadingMode !== 'ST') continue;
    for (const result of config.results) if (result.testId === 'd') {
      result.status = 'failed';
      result.durationSeconds = null;
      result.error = 'Fictional unanchored failure';
    }
  }
  return validatePublishedDashboardData(source).data;
}
const scope = { targets: ['gfx1250', 'gfx950'], modes: ['ST', 'MT'], suites: ['Triton', 'PyTorch'] };

function withoutUnanchoredMembership(data) {
  const expected = structuredClone(data);
  for (const catalog of Object.values(expected.catalogs)) {
    catalog.configurations['gfx1250:ST'] = catalog.configurations['gfx1250:ST']?.filter((id) => id !== 'd');
  }
  return expected;
}

test('unanchored additions are excluded from every normalized point but not raw failures or comparisons', () => {
  const data = unanchoredData();
  const filters = { ...scope, suites: data.suites };
  const before = structuredClone(data);
  const actual = selectOverview(data, filters);
  const expected = selectOverview(withoutUnanchoredMembership(data), filters);
  expect(actual.history.series).toEqual(expected.history.series);
  expect(actual.history.currentDuration).toBe(expected.history.currentDuration);
  expect(actual.history.durationDelta).toBe(expected.history.durationDelta);
  expect(actual.metrics.duration).toBe(expected.metrics.duration);
  expect(actual.metrics.durationDelta).toBe(expected.metrics.durationDelta);
  expect(actual.metrics.failed).toBeGreaterThan(0);
  expect(actual.results.some((result) => result.testId === 'gfx1250:ST:d' && result.status === 'failed')).toBe(true);
  expect(actual.history.anchors.some((anchor) => anchor.testId === 'gfx950:ST:d')).toBe(true);
  expect(actual.history.anchors.some((anchor) => anchor.testId === 'gfx1250:ST:d')).toBe(false);
  expect(selectRunComparison(actual.candidate, actual.baseline, filters).notComparable.some(({ candidateTest }) => candidateTest?.testId === 'gfx1250:ST:d')).toBe(true);
  expect(data).toEqual(before);
});

test('the first successful anchor includes the addition consistently in older and current normalized totals', () => {
  const data = unanchoredData();
  const filters = { ...scope, targets: ['gfx1250'], modes: ['ST'], suites: data.suites };
  const before = selectOverview(data, filters);
  const latest = data.latestCommitRun;
  const addition = latest.tests.find(({ logicalTestId, mode, target }) => logicalTestId === 'd' && mode === 'ST' && target === 'gfx1250');
  addition.status = 'completed';
  addition.durationSeconds = 17;
  const after = selectOverview(data, filters);
  expect(after.history.series[0].data[0]).toBeCloseTo(before.history.series[0].data[0] + 17);
  expect(after.metrics.duration).toBeCloseTo(before.metrics.duration + 17);
  expect(after.history.anchors.find(({ testId }) => testId === 'gfx1250:ST:d')).toMatchObject({ runId: latest.runId, durationSeconds: 17 });
  const failedIndex = after.history.slots.findIndex(({ run }) => run?.runId === 'fictional-develop-11');
  expect(after.history.series[0].data[failedIndex]).toBeNull();
});

test('an empty eligible workload is unavailable rather than a measured zero', () => {
  const data = unanchoredData();
  for (const catalog of Object.values(data.catalogs)) catalog.configurations['gfx1250:ST'] = ['d'];
  const overview = selectOverview(data, { ...scope, targets: ['gfx1250'], modes: ['ST'], suites: data.suites });
  expect(overview.history.series[0].data.every((value) => value === null)).toBe(true);
  expect(overview.metrics.duration).toBeNull();
  expect(overview.metrics.durationDelta).toBeNull();
});
