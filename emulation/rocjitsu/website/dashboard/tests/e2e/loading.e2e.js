import { expect, test } from '@playwright/test';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
import { downloadSource, fetchPolicies, installPublication, openDashboard, ready, recordFetchPolicies } from './helpers/dashboard.js';

test('loading shell reports partial progress and disables data actions until all indexed runs settle', async ({ page }) => {
  let release;
  let held = false;
  const gate = new Promise((resolve) => { release = resolve; });
  await installPublication(page, { beforeResponse: async ({ relative }) => {
    // A permanent handler and synchronous flag avoid interception-disable races.
    if (relative.startsWith('runs/') && !held) {
      held = true;
      await gate;
    }
  } });
  await page.goto('/');
  try {
    await expect.poll(() => held).toBe(true);
    const loading = page.getByTestId('dashboard-data-loading');
    await expect(loading).toHaveAttribute('aria-busy', 'true');
    await expect(loading.getByRole('status')).toHaveText('Loading benchmark run data');
    const progress = loading.getByRole('progressbar', { name: 'Loading benchmark run data' });
    await expect.poll(async () => Number(await progress.getAttribute('aria-valuenow'))).toBeGreaterThan(0);
    expect(Number(await progress.getAttribute('aria-valuenow'))).toBeLessThan(100);
    await expect(progress).toHaveAttribute('aria-valuetext', /\d+ of 44 run files loaded/);
    await expect(page.getByRole('button', { name: 'Download JSON' })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Reload all data' })).toBeDisabled();
    await expect(page.getByTestId('dashboard-navigation').getByRole('tab')).toHaveCount(4);
    await page.getByRole('tab', { name: 'Benchmarks', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Benchmarks', level: 1, exact: true })).toBeVisible();
  } finally {
    release();
  }
  await ready(page);
  await expect(page.getByTestId('benchmark-grid-card')).toHaveCount(4);
});

test('fatal load and failed refresh discard stale exports; Retry restores raw schema 2 and cache generations', async ({ page }) => {
  let status = 404;
  let indexFailures = 0;
  const { publication } = await installPublication(page, { beforeResponse: ({ relative }) => {
    if (relative === 'index.json' && status !== 200) {
      indexFailures += 1;
      return { status, contentType: 'text/plain', body: 'Fictional endpoint unavailable' };
    }
  } });
  await recordFetchPolicies(page);
  await page.goto('/');
  const failure = page.getByTestId('dashboard-data-error');
  await expect(failure).toContainText('No available test data');
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeDisabled();
  for (const name of ['Branch Runs', 'Benchmarks', 'Run Comparison', 'Overview']) {
    await page.getByRole('tab', { name, exact: true }).click();
    await expect(failure).toBeVisible();
  }
  await expect(page.getByRole('img')).toHaveCount(0);
  status = 200;
  await failure.getByRole('button', { name: 'Retry', exact: true }).click();
  await ready(page);
  const source = await downloadSource(page);
  expect(source).toEqual(publication);
  expect(source.metadata).toMatchObject({ schemaVersion: 2, isBeta: true });
  expect(source.runs.every((run) => !('plugin' in run) && !('comparisonId' in run))).toBe(true);
  expect(source.runs.some((run) => run.source.branch.startsWith('fictional/'))).toBe(true);
  const policies = await fetchPolicies(page);
  const initialImmutable = policies.filter(({ url }) => /\/(runs|test-catalogs)\//.test(url));
  expect(initialImmutable).toHaveLength(publication.runs.length + Object.keys(publication.catalogs).length);
  expect(policies.filter(({ url }) => /\/(metadata|index)\.json/.test(url)).every(({ cache }) => cache === 'no-store')).toBe(true);
  expect(initialImmutable.every(({ cache }) => cache === 'force-cache')).toBe(true);

  const before = indexFailures;
  status = 503;
  await page.getByRole('button', { name: 'Reload all data' }).click();
  await expect(failure).toContainText('Unable to reach published dashboard data');
  expect(indexFailures - before).toBe(3);
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Reload all data' })).toBeEnabled();
  await expect(page.getByTestId('recent-runs-table').locator('tbody tr')).toHaveCount(0);
  status = 200;
  await page.evaluate(() => { window.__dashboardFetches = []; });
  await failure.getByRole('button', { name: 'Retry', exact: true }).click();
  await ready(page);
  const generation = await page.evaluate(() => localStorage.getItem('rocjitsu-data-cache-generation'));
  expect(generation).toBeTruthy();
  const refreshed = await fetchPolicies(page);
  expect(refreshed.length).toBeGreaterThan(0);
  expect(refreshed.every(({ url }) => new URL(url).searchParams.get('reload') === generation)).toBe(true);
  expect(refreshed.filter(({ url }) => /\/(runs|test-catalogs)\//.test(url)).every(({ cache }) => cache === 'reload')).toBe(true);
  expect(await downloadSource(page)).toEqual(source);
  await page.reload();
  await ready(page);
  const normal = await fetchPolicies(page);
  const immutable = normal.filter(({ url }) => /\/(runs|test-catalogs)\//.test(url));
  expect(immutable.length).toBeGreaterThan(0);
  expect(immutable.every(({ url, cache }) => cache === 'force-cache' && new URL(url).searchParams.get('reload') === generation)).toBe(true);
  expect(normal.filter(({ url }) => /\/(metadata|index)\.json/.test(url)).every(({ url, cache }) => cache === 'no-store' && !new URL(url).searchParams.has('reload'))).toBe(true);
});

test('one invalid indexed run fails the entire publication instead of exposing partial measurements', async ({ page }) => {
  const publication = createSchema2Publication();
  publication.runs.find((run) => run.id === 'fictional-develop-23').configurations[0].results[0].durationSeconds = -1;
  await installPublication(page, { publication });
  await page.goto('/');
  await expect(page.getByTestId('dashboard-data-error')).toContainText('fictional-develop-23');
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeDisabled();
  await expect(page.getByTestId('metric-card-run-health')).toContainText('No selected results available');
  await expect(page.getByRole('img')).toHaveCount(0);
  await page.getByRole('tab', { name: 'Branch Runs', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'No published branch runs' })).toBeVisible();
  await expect(page.getByTestId('branch-config-gfx1250-ST')).toHaveCount(0);
});

test('optional storage failure cannot discard a successfully refreshed publication', async ({ page }) => {
  await page.addInitScript(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function setItem(key, value) {
      if (key === 'rocjitsu-data-cache-generation') throw new DOMException('Fictional quota failure', 'QuotaExceededError');
      return original.call(this, key, value);
    };
  });
  await openDashboard(page);
  await page.getByRole('button', { name: 'Reload all data' }).click();
  await ready(page);
  expect((await downloadSource(page)).metadata.schemaVersion).toBe(2);
});
