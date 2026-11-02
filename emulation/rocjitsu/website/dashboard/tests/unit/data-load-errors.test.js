import { expect, test } from 'vitest';
import {
  MAX_DISPLAYED_VALIDATION_ERRORS,
  summarizeDashboardDataError,
} from '../../src/data/dashboardDataError.js';

test('limits validation details and reports how many were omitted', () => {
  const details = Array.from({ length: 13 }, (_, index) => `- runs/run-${index}.json: invalid`);
  const error = new Error(`Dashboard data failed validation:\n${details.join('\n')}`);

  const summary = summarizeDashboardDataError(error);

  expect(summary).toContain(details[MAX_DISPLAYED_VALIDATION_ERRORS - 1]);
  expect(summary).not.toContain(details[MAX_DISPLAYED_VALIDATION_ERRORS]);
  expect(summary).toContain('3 additional validation errors omitted');
});

test('does not alter short validation or fetch errors', () => {
  const validationError = new Error('Dashboard data failed validation:\n- one\n- two');
  const fetchError = new Error('Unable to load dashboard data index (503)');

  expect(summarizeDashboardDataError(validationError)).toBe(validationError.message);
  expect(summarizeDashboardDataError(fetchError)).toBe(fetchError.message);
});

test('classifies unavailable, missing, and invalid published data', () => {
  const unavailable = new Error(
    'Unable to reach dashboard data index https://raw.githubusercontent.com/example/index.json',
  );
  unavailable.dashboardDataErrorCode = 'unavailable';
  const missing = new Error('Unable to load dashboard data index (404)');
  missing.dashboardDataErrorCode = 'missing';
  const invalid = new Error('Unable to parse dashboard data index as JSON');
  invalid.dashboardDataErrorCode = 'invalid';

  expect(summarizeDashboardDataError(unavailable)).toContain(
    'Unable to reach published dashboard data on GitHub Raw',
  );
  expect(summarizeDashboardDataError(missing)).toBe('No published test data is available.');
  expect(summarizeDashboardDataError(invalid)).toContain('Published test data is invalid.');
});
