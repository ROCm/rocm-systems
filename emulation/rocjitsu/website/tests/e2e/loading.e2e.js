import { expect, test } from '@playwright/test';

test('loads the data-driven overview without browser errors', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });

  await page.goto('/');
  await expect(page.getByTestId('rocjitsu-logo')).toHaveCount(0);
  await expect(page.getByText('Beta', { exact: true })).toBeVisible();
  await expect(page.getByText('Demo', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('heading', { name: 'RocJitsu Performance Health' })).toBeVisible();
  await expect(page.getByText('Performance Trend')).toBeVisible();
  await expect(page.getByText('Overview run coverage', { exact: false })).toBeVisible();
  await expect(page.getByText('Recent Runs')).toBeVisible();
  await expect(page.getByText('RocJitsu Commit Activity')).toHaveCount(0);
  await expect(page.locator('canvas')).toHaveCount(1);
  await expect(page.getByText('GEMM BF16 4096³').first()).toBeVisible();
  expect(await page.evaluate(() => Object.hasOwn(window, 'ROCJITSU_BENCHMARK_DATA'))).toBe(false);
  expect(errors).toEqual([]);
});

test('renders the dashboard shell and run progress while data is still loading', async ({ page }) => {
  let releaseRunRequest;
  const runRequestGate = new Promise((resolve) => {
    releaseRunRequest = resolve;
  });
  await page.route('**/data/runs/**', async (route) => {
    await runRequestGate;
    await route.continue();
  }, { times: 1 });

  await page.goto('/');
  try {
    await expect(page.getByText('RocJitsu / Performance Dashboard')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'RocJitsu Performance Health' })).toBeVisible();
    await expect(page.getByText('Beta', { exact: true })).toBeVisible();
    const loadingState = page.getByTestId('dashboard-data-loading');
    await expect(loadingState).toBeVisible();
    await expect(loadingState).toHaveAttribute('aria-busy', 'true');
    await expect(loadingState.getByRole('status')).toHaveText('Loading benchmark run data');
    const progress = loadingState.getByRole('progressbar', { name: 'Loading benchmark run data' });
    await expect(progress).toBeVisible();
    await expect(loadingState.getByTestId('dashboard-load-progress')).toContainText(/\d+ of \d+ run files/);
    await expect.poll(() => progress.getAttribute('aria-valuenow')).not.toBeNull();
    await expect(progress).toHaveAttribute('aria-valuetext', /\d+ of \d+ run files loaded/);
    await expect(page.getByRole('button', { name: 'Download JSON' })).toBeDisabled();
  } finally {
    releaseRunRequest();
  }

  await expect(page.getByTestId('dashboard-data-loading')).toHaveCount(0);
  await expect(page.getByTestId('dashboard-navigation')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeEnabled();
});

test('renders valid history with a warning when an indexed run is invalid', async ({ page }) => {
  const invalidRunFile = 'runs/invalid-run.json';
  await page.route('**/data/index.json', async (route) => {
    const response = await route.fetch();
    const index = await response.json();
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ...index, runFiles: [index.runFiles[0], invalidRunFile] }),
    });
  });
  await page.route('**/data/runs/invalid-run.json', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{"id":"invalid-run"}' });
  });

  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'RocJitsu Performance Health' })).toBeVisible();
  const warning = page.getByTestId('invalid-run-warning');
  await expect(warning).toContainText('Skipped 1 invalid run file');
  await expect(warning).toContainText(invalidRunFile);
  await expect(warning).toContainText('references an invalid test catalog');
});

test('offers a working Retry after a fatal data failure', async ({ page }) => {
  let failIndex = true;
  await page.route('**/data/index.json', async (route) => {
    if (!failIndex) {
      await route.continue();
      return;
    }
    failIndex = false;
    await route.fulfill({ status: 503, contentType: 'text/plain', body: 'unavailable' });
  });

  await page.goto('/');
  const failure = page.getByTestId('dashboard-data-error');
  await expect(failure).toContainText('Dashboard data unavailable');
  await expect(failure).toContainText('503');

  await failure.getByRole('button', { name: 'Retry' }).click();

  await expect(page.getByTestId('dashboard-data-error')).toHaveCount(0);
  await expect(page.getByTestId('dashboard-navigation')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'RocJitsu Performance Health' })).toBeVisible();
});
