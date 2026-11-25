import { readFile } from 'node:fs/promises';
import { expect } from '@playwright/test';

export const RAW_DATA_PREFIX = 'https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/';

export async function checkPublishedData(page, testInfo, { timeout = 60_000 } = {}) {
  const pageErrors = [];
  const dialogs = [];
  const runResponses = [];
  const runRequests = [];
  const localDataRequests = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  page.on('dialog', async (dialog) => {
    dialogs.push(`${dialog.type()}: ${dialog.message()}`);
    await dialog.dismiss();
  });
  page.on('response', (response) => {
    if (response.url().startsWith(`${RAW_DATA_PREFIX}runs/`)) {
      runResponses.push({ url: response.url(), ok: response.ok() });
    }
  });
  page.on('request', (request) => {
    if (request.url().startsWith(`${RAW_DATA_PREFIX}runs/`)) runRequests.push(request.url());
    const url = new URL(request.url());
    if (url.origin === new URL(testInfo.project.use.baseURL).origin && url.pathname.startsWith('/data/')) {
      localDataRequests.push(url.href);
    }
  });

  // Observe the same real production UI in both live and deterministic boundary checks.
  try {
    const indexResponsePending = page.waitForResponse(`${RAW_DATA_PREFIX}index.json`, { timeout });
    await page.goto('/');
    const indexResponse = await indexResponsePending;
    const missing = indexResponse.status() === 404;
    let index;
    if (!missing) {
      expect(indexResponse.ok(), 'Published index HTTP status').toBe(true);
      index = await indexResponse.json();
      await testInfo.attach('published-index', { body: JSON.stringify(index, null, 2), contentType: 'application/json' });
      expect(Array.isArray(index?.runFiles), 'Published index must list run files').toBe(true);
    }

    const loadError = page.getByTestId('dashboard-data-error');
    const download = page.getByRole('button', { name: 'Download JSON', exact: true });
    // Wait for a terminal outcome, not an absent alert before fetch/adaptation settles.
    await expect.poll(async () => {
      if (await loadError.count()) return `load error: ${await loadError.innerText()}`;
      return await download.isEnabled() ? 'loaded' : 'loading';
    }, { timeout, message: 'Published data must reach a terminal load state' }).not.toBe('loading');
    await expect(page.getByTestId('dashboard-data-loading')).toHaveCount(0);
    if (missing) {
      // Only the specific missing-data banner is expected, not arbitrary loader errors.
      await expect(loadError).toBeVisible();
      await expect(loadError).toHaveText(/^No available test data\s*No published test data is available\.\s*Dashboard values will remain empty until benchmark data is published to the site\.\s*Retry$/);
      await expect(download).toBeDisabled();
      expect(runRequests, 'Missing publication must not request runs').toEqual([]);
      return;
    }
    await expect(download, await loadError.count() ? await loadError.innerText() : 'Published data must load successfully').toBeEnabled();
    await expect(loadError).toHaveCount(0);
    await expect(page.getByRole('alert')).toHaveCount(0);

    const downloadPending = page.waitForEvent('download');
    await download.click();
    const source = JSON.parse(await readFile(await (await downloadPending).path(), 'utf8'));
    expect(source.index).toEqual(index);
    expect(source.runs.length).toBe(index.runFiles.length);
    if (index.runFiles.length > 0) {
      expect(source.runs.length).toBeGreaterThan(0);
    } else {
      expect(runRequests, 'Empty publication must not request runs').toEqual([]);
    }
    expect([...new Set(runResponses.map(({ url }) => url))].sort())
      .toEqual(index.runFiles.map((file) => `${RAW_DATA_PREFIX}${file}`).sort());
    expect(runResponses.every(({ ok }) => ok), 'All published run requests must succeed').toBe(true);
  } finally {
    expect(pageErrors, 'Uncaught browser page errors').toEqual([]);
    expect(dialogs, 'Unexpected JavaScript dialogs').toEqual([]);
    expect(localDataRequests, 'Production must not substitute local data').toEqual([]);
    await expect(page.getByTestId('fixture-data-warning')).toHaveCount(0);
    await expect(page.locator('[role="alert"]:not([data-testid="dashboard-data-error"])')).toHaveCount(0);
  }
}
