import { expect, test } from 'vitest';
import {
  LOCAL_DATA_BASE_URL,
  PUBLISHED_DATA_BASE_URL,
  resolveDataBaseUrl,
} from '../../scripts/dashboard-data-source.mjs';

test('fetches the dashboard data directory on the data branch by default', () => {
  expect(resolveDataBaseUrl('production', {})).toBe(PUBLISHED_DATA_BASE_URL);
});

test('serves development, fixtures, and mounted local data from the application', () => {
  expect(resolveDataBaseUrl('development', {})).toBe(LOCAL_DATA_BASE_URL);
  expect(resolveDataBaseUrl('fixtures', {})).toBe(LOCAL_DATA_BASE_URL);
  expect(resolveDataBaseUrl('production', {
    DASHBOARD_DATA_DIR: '/tmp/rocjitsu-data',
  })).toBe(LOCAL_DATA_BASE_URL);
});
