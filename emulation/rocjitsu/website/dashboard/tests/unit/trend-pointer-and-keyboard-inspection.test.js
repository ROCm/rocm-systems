import { expect, test } from 'vitest';
import { nearestTrendIndex, trendEmptyMessage, trendKeyIndex, trendSlotStatus } from '../../src/components/overview/overviewPresentation.js';

const comparison = {
  range: '1M', slots: [{ dayKey: '2026-01-01', run: null }, { dayKey: '2026-01-02', run: { runId: 'failed-attempt', timestamp: '2026-01-02T12:00:00Z' } }],
  attempts: [{ eligible: false }], rows: [{ target: 'gfx950', mode: 'single', cohortIds: ['a'], points: [{ reason: 'no-run', speedup: null }, { reason: 'timeout', speedup: null }] }],
};
test('keyboard gap status identifies the actual slot and reason without borrowing a measurement', () => {
  expect(trendSlotStatus(comparison, 0)).toContain('2026-01-01 UTC');
  expect(trendSlotStatus(comparison, 0)).toContain('No published run in this view');
  expect(trendSlotStatus(comparison, 1)).toContain('failed-attempt');
  expect(trendSlotStatus(comparison, 1)).toContain('Timed out');
  expect(trendSlotStatus({ ...comparison, range: '1W' }, 0)).toContain('No eligible plotted run');
});
test('empty comparison copy separates scope, candidates, shared cohort and weekly eligibility', () => {
  expect(trendEmptyMessage({ ...comparison, rows: [] })).toContain('Select targets');
  expect(trendEmptyMessage({ ...comparison, attempts: [] })).toBe('No develop runs in the selected range/benchmark set.');
  expect(trendEmptyMessage({ ...comparison, rows: [{ ...comparison.rows[0], cohortIds: [] }] })).toContain('No shared workloads');
  expect(trendEmptyMessage({ ...comparison, range: '1W' })).toBe('No complete runs for selected comparison in this week.');
  expect(trendEmptyMessage(comparison)).toContain('no positive valid reference');
});

test('magnetic column resolution considers the nearest slot before measurement eligibility', () => {
  const projection = { slots: [{ x: 0, run: {} }, { x: 1, run: {} }, { x: 2, run: {} }],
    rows: [{ points: [{ speedup: 1 }, { speedup: null }, { speedup: 1.2 }] },
      { points: [{ speedup: null }, { speedup: null }, { speedup: 0.9 }] }] };
  expect(nearestTrendIndex(projection, 1)).toBeNull();
  expect(nearestTrendIndex(projection, 2)).toBe(2);
  expect(nearestTrendIndex(projection, NaN)).toBeNull();
});
test('keyboard navigation visits every supplied slot and clamps first/last/clear', () => {
  expect(trendKeyIndex([0, 1, 2], null, 'ArrowRight')).toBe(0);
  expect(trendKeyIndex([0, 1, 2], null, 'ArrowLeft')).toBe(2);
  expect(trendKeyIndex([0, 1, 2], 0, 'ArrowRight')).toBe(1);
  expect(trendKeyIndex([0, 1, 2], 1, 'ArrowRight')).toBe(2);
  expect(trendKeyIndex([0, 1, 2], 1, 'ArrowLeft')).toBe(0);
  expect(trendKeyIndex([0, 1, 2], 0, 'ArrowLeft')).toBe(0);
  expect(trendKeyIndex([0, 1, 2], 2, 'ArrowRight')).toBe(2);
  expect(trendKeyIndex([0, 1, 2], null, 'End')).toBe(2);
  expect(trendKeyIndex([0, 1, 2], 2, 'Home')).toBe(0);
  expect(trendKeyIndex([0, 1, 2], 1, 'Escape')).toBeNull();
});
