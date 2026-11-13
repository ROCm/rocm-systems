import { expect, test } from 'vitest';
import { formatDuration } from '../../src/utils/formatters.js';

test.each([
  { value: 119.99, expected: '2m 0.0s' },
  { value: 179.99, expected: '3m 0.0s' },
  { value: 119.94, expected: '1m 59.9s' },
  { value: 60, expected: '1m 0.0s' },
  { value: 61.25, expected: '1m 1.3s' },
  { value: 0, expected: '0.00s' },
  { value: null, expected: '—' },
  { value: Infinity, expected: '—' },
])('duration $value displays $expected without a 60-second remainder', ({ value, expected }) => {
  expect(formatDuration(value)).toBe(expected);
});
