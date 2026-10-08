import { expect, test } from 'vitest';
import * as presentation from '../../src/components/overview/overviewPresentation.js';
import { selectRecentRuns } from '../../src/data/selectors.js';
import { createRecentRunsHistory } from '../fixtures/recent-runs-history.js';

const { data } = createRecentRunsHistory();
const filters = { targets: ['gfx1250'], suites: ['Triton'], modes: ['ST'] };

test('page bounds clamp shrinking, short and empty histories without overlapping ranges', () => {
  expect(presentation.recentRunsPage?.(65, 0)).toEqual({ page: 0, start: 0, end: 20 });
  expect(presentation.recentRunsPage?.(65, 1)).toEqual({ page: 1, start: 20, end: 40 });
  expect(presentation.recentRunsPage?.(65, 2)).toEqual({ page: 2, start: 40, end: 60 });
  expect(presentation.recentRunsPage?.(65, 3)).toEqual({ page: 3, start: 60, end: 65 });
  expect(presentation.recentRunsPage?.(24, 3)).toEqual({ page: 1, start: 20, end: 24 });
  expect(presentation.recentRunsPage?.(3, 3)).toEqual({ page: 0, start: 0, end: 3 });
  expect(presentation.recentRunsPage?.(0, 3)).toEqual({ page: 0, start: 0, end: 0 });
});

test('uncapped history preserves validated canonical attempts, tied execution order and selected coverage', () => {
  const rows = selectRecentRuns(data, filters, Infinity);
  expect(rows).toHaveLength(65);
  expect(rows.map(({ run }) => run.runId)).toEqual(Array.from({ length: 65 }, (_, index) => `fictional-pagination-${String(64 - index).padStart(3, '0')}`));
  for (const row of rows) {
    expect(row.run.branch).toBe('develop');
    const tests = row.run.tests.filter((test) => filters.targets.includes(test.target) && filters.modes.includes(test.mode) && filters.suites.includes(test.suite));
    expect(row.total).toBe(tests.length);
    expect(row.completed).toBe(tests.filter(({ status }) => status === 'completed').length);
  }
  expect(rows.find(({ run }) => run.runId === 'fictional-pagination-008').failed).toBeGreaterThan(0);
  for (const key of ['targets', 'suites', 'modes']) {
    const emptyScope = selectRecentRuns(data, { ...filters, [key]: [] }, Infinity);
    expect(emptyScope).toHaveLength(65);
    expect(emptyScope.every((row) => row.total === 0 && row.duration === null)).toBe(true);
  }
});
