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
  await expect(page.getByRole('heading', { name: 'Rocjitsu Simulation Performance' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Performance Trend' })).toBeVisible();
  await expect(page.getByText('Run coverage', { exact: true })).toBeVisible();
  await expect(page.getByText('Recent Runs')).toBeVisible();
  await expect(page.getByText('Rocjitsu Commit Activity')).toHaveCount(0);
  await expect(page.locator('canvas')).toHaveCount(1);
  await expect(page.getByText('GEMM BF16 4096³').first()).toBeVisible();
  expect(await page.evaluate(() => Object.hasOwn(window, 'ROCjITSU_BENCHMARK_DATA'))).toBe(false);
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
    await expect(page.getByText('Rocjitsu / Simulation Performance Dashboard')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Rocjitsu Simulation Performance' })).toBeVisible();
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
    await expect(page.getByRole('button', { name: 'Reload all data' })).toBeDisabled();
  } finally {
    releaseRunRequest();
  }

  await expect(page.getByTestId('dashboard-data-loading')).toHaveCount(0);
  await expect(page.getByTestId('dashboard-navigation')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeEnabled();
});

test('offers an explicit full data reload', async ({ page }) => {
  let indexRequests = 0;
  let cacheBustedIndexRequests = 0;
  const runGenerations = [];
  await page.route('**/data/index.json*', async (route) => {
    indexRequests += 1;
    if (new URL(route.request().url()).searchParams.has('reload')) {
      cacheBustedIndexRequests += 1;
    }
    await route.continue();
  });
  await page.route('**/data/runs/*.json*', async (route) => {
    runGenerations.push(new URL(route.request().url()).searchParams.get('reload'));
    await route.continue();
  });

  await page.goto('/');
  const reload = page.getByRole('button', { name: 'Reload all data' });
  await expect(reload).toBeEnabled();
  await expect(reload).toHaveClass(/MuiButton-colorInherit/);
  const initialIndexRequests = indexRequests;
  await reload.click();

  await expect.poll(() => indexRequests).toBeGreaterThan(initialIndexRequests);
  expect(cacheBustedIndexRequests).toBeGreaterThan(0);
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeEnabled();

  const savedGeneration = await page.evaluate(
    () => window.localStorage.getItem('rocjitsu-data-cache-generation'),
  );
  expect(savedGeneration).toBeTruthy();

  runGenerations.length = 0;
  await page.reload();
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeEnabled();
  expect(runGenerations.length).toBeGreaterThan(0);
  expect(runGenerations.every((generation) => generation === savedGeneration)).toBe(true);
});

test('keeps loaded data when cache generation persistence fails', async ({ page }) => {
  await page.addInitScript(() => {
    const originalSetItem = Storage.prototype.setItem;
    Storage.prototype.setItem = function setItem(key, value) {
      if (key === 'rocjitsu-data-cache-generation') {
        throw new DOMException('Quota exceeded', 'QuotaExceededError');
      }
      return originalSetItem.call(this, key, value);
    };
  });

  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeEnabled();

  await page.getByRole('button', { name: 'Reload all data' }).click();

  await expect(page.getByTestId('dashboard-data-error')).toHaveCount(0);
  await expect(page.getByTestId('dashboard-navigation')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeEnabled();
});

test('fails closed when an indexed run is invalid', async ({ page }) => {
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

  const failure = page.getByTestId('dashboard-data-error');
  await expect(failure).toContainText('No available test data');
  await expect(failure).toContainText(invalidRunFile);
  await expect(failure).toContainText('references an invalid test catalog');
  await expect(page.getByRole('button', { name: 'Reload all data' }))
    .toHaveClass(/MuiButton-colorPrimary/);
  await expect(page.getByTestId('latest-results')).toBeVisible();
  await expect(page.getByTestId('latest-results').locator('tbody tr')).toHaveCount(1);
});

test('offers a working Retry after a fatal data failure', async ({ page }) => {
  let remainingFailures = 3;
  await page.route('**/data/index.json', async (route) => {
    if (remainingFailures === 0) {
      await route.continue();
      return;
    }
    remainingFailures -= 1;
    await route.abort('failed');
  });

  await page.goto('/');
  const failure = page.getByTestId('dashboard-data-error');
  await expect(failure).toContainText('No available test data');
  await expect(failure).toContainText('Unable to reach published dashboard data');
  await expect(page.getByTestId('dashboard-navigation')).toBeVisible();
  await expect(page.getByLabel('Targets')).toBeDisabled();
  await expect(page.getByLabel('Suites')).toBeDisabled();
  await expect(page.getByText('Total duration', { exact: true })).toBeVisible();
  await expect(page.getByText('Run health', { exact: true })).toBeVisible();
  await expect(page.getByText('Performance Trend')).toBeVisible();
  await expect(page.getByText('Largest Changes')).toBeVisible();
  await expect(page.getByText('Latest Commit Results')).toBeVisible();
  await expect(page.getByText('Recent Runs')).toBeVisible();
  await expect(page.getByTestId('latest-commit-run').getByText('—', { exact: true })).toHaveCount(2);

  await page.getByRole('tab', { name: 'Benchmarks' }).click();
  await expect(page.getByText('Benchmark Explorer')).toBeVisible();
  await expect(page.getByText('Benchmark Run History')).toBeVisible();
  await expect(page.getByText('Seconds', { exact: true })).toHaveCount(0);
  await page.getByRole('tab', { name: 'Run Comparison' }).click();
  await expect(page.getByText('Performance Change by Benchmark')).toBeVisible();
  await page.getByRole('tab', { name: 'Plugin Comparison' }).click();
  await expect(page.getByTestId('plugin-comparison-empty')).toContainText('No vanilla baseline or sanitizer comparison runs are available');
  await expect(page.getByText('Per-Test Runtime Overhead')).toBeVisible();
  await expect(page.getByText('Plugin Errors')).toBeVisible();
  await page.getByRole('tab', { name: 'Failures' }).click();
  await expect(page.getByText('Run Reliability')).toBeVisible();
  await expect(page.getByText('Failed and Timed-Out Cases')).toBeVisible();
  await expect(page.getByText('101%', { exact: true })).toHaveCount(0);
  await expect(page.getByTestId('failure-range')).toHaveCount(0);

  await failure.getByRole('button', { name: 'Retry' }).click();

  await expect(page.getByTestId('dashboard-data-error')).toHaveCount(0);
  await expect(page.getByTestId('dashboard-navigation')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Rocjitsu Simulation Performance' })).toBeVisible();
});
