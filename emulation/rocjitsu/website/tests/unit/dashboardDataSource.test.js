import { expect, test } from 'vitest';
import {
  LOCAL_DATA_BASE_URL,
  resolveDataBaseUrl,
} from '../../scripts/dashboard-data-source.mjs';

test('serves data next to the application outside the pages build mode', () => {
  for (const mode of ['development', 'production', 'fixtures']) {
    expect(resolveDataBaseUrl(mode, {})).toBe(LOCAL_DATA_BASE_URL);
  }
});

test('fetches the dashboard data directory on the data branch in pages mode', () => {
  expect(resolveDataBaseUrl('pages', { GITHUB_REPOSITORY: 'example/rocm-systems' })).toBe(
    'https://raw.githubusercontent.com/example/rocm-systems/refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/',
  );
});

test('prefers an explicit base URL in every mode', () => {
  const env = { VITE_DASHBOARD_DATA_BASE_URL: 'https://data.example/rocjitsu/' };
  expect(resolveDataBaseUrl('production', env)).toBe(env.VITE_DASHBOARD_DATA_BASE_URL);
  expect(resolveDataBaseUrl('pages', env)).toBe(env.VITE_DASHBOARD_DATA_BASE_URL);
});

test('rejects a pages build that cannot name its data branch', () => {
  expect(() => resolveDataBaseUrl('pages', {})).toThrow(/GITHUB_REPOSITORY/);
  expect(() => resolveDataBaseUrl('pages', { GITHUB_REPOSITORY: 'rocm-systems' }))
    .toThrow(/<owner>\/<repository>/);
});
