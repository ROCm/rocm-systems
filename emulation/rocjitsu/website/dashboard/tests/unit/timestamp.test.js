import { expect, test } from 'vitest';
import { isValidTimestamp } from '../../src/utils/timestamp.js';
import { formatCompactRunTime, formatFullDate, formatRunTime, formatShortDate } from '../../src/utils/formatters.js';

test.each(['2026-10-05T14:00:00Z', '2026-10-05', '2026-10-05T23:07:00-04:00'])('%s is a usable timestamp', (value) => {
  expect(isValidTimestamp(value)).toBe(true);
});

test.each([null, undefined, '', 'not a date', '2026-99-99', 0, 1790000000000, true, {}, []])('%j is not a usable timestamp', (value) => {
  expect(isValidTimestamp(value)).toBe(false);
});

test('every formatter shows its own missing mark for an unusable timestamp, and never a date', () => {
  for (const value of [null, '', 'invalid', 1790000000000]) {
    expect(formatShortDate(value)).toBe('-');
    expect(formatFullDate(value)).toBe('-');
    expect(formatCompactRunTime(value)).toBe('-');
    expect(formatRunTime(value)).toBe('—');
  }
});
