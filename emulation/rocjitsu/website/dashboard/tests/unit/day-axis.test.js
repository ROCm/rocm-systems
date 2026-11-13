import { expect, test } from 'vitest';
import { dayAxisBounds, dayXAxis } from '../../src/utils/dayAxis.js';

const dayKeys = ['2026-10-01', '2026-10-02', '2026-10-03'];
const theme = { palette: { text: { secondary: '#555' }, divider: '#ddd' } };

test('1M and 3M always start at the first day, whatever slot carries the first data', () => {
  const slots = [{ x: 0 }, { x: 1 }, { x: 2 }];
  for (const range of ['1M', '3M']) expect(dayAxisBounds({ range, slots, dayKeys, axisMax: 2 }, 2).firstX).toBe(0);
});

test('ALL starts at the first slot with data, or at 0 when none has any', () => {
  const slots = [{ x: 0 }, { x: 1 }, { x: 2 }];
  expect(dayAxisBounds({ range: 'ALL', slots, dayKeys, axisMax: 2 }, 1).firstX).toBe(1);
  expect(dayAxisBounds({ range: 'ALL', slots, dayKeys, axisMax: 2 }, null).firstX).toBe(0);
});

test('1W spreads several runs a day inside the day and keeps the axis at least one day wide', () => {
  const slots = [{ x: 0.33 }, { x: 0.67 }, { x: 1.5 }];
  const bounds = dayAxisBounds({ range: '1W', slots, dayKeys, axisMax: 3 }, 0);
  expect(bounds).toMatchObject({ weekly: true, firstX: 0.33, lastX: 2.5 });
  const oneDay = dayAxisBounds({ range: '1W', slots: [{ x: 0.5 }], dayKeys: ['2026-10-01'], axisMax: 1 }, 0);
  expect(oneDay.lastX - oneDay.firstX).toBeGreaterThanOrEqual(1);
});

test('an empty range still yields a valid axis', () => {
  const bounds = dayAxisBounds({ range: 'ALL', slots: [], dayKeys: [], axisMax: 0 }, null);
  expect(bounds.lastX).toBeGreaterThan(bounds.firstX);
  const axis = dayXAxis(bounds, theme);
  expect(axis.min).toBe(0);
  expect(axis.axisLabel.formatter(0)).toBe('');
});

test('labels name the day of the slot: weekly by floor, others by rounding', () => {
  const weekly = dayXAxis({ weekly: true, firstX: 0, lastX: 3, dateStep: 1, dayKeys }, theme).axisLabel;
  expect(weekly.customValues).toEqual([0.5, 1.5, 2.5]);
  expect(weekly.formatter(1.5)).toBe('Oct 2');
  const range = dayXAxis({ weekly: false, firstX: 0, lastX: 2, dateStep: 1, dayKeys }, theme).axisLabel;
  expect(range.formatter(2)).toBe('Oct 3');
});
