import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';

const RAW_DATA_PREFIX = 'https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/';

test('new production build loads the existing published index and runs without browser errors', async ({ page }, testInfo) => {
  test.skip(process.env.PLAYWRIGHT_PUBLISHED_DATA !== '1', 'Opt in to live network coverage with PLAYWRIGHT_PUBLISHED_DATA=1');
  test.setTimeout(90_000);
  const pageErrors = [];
  const dialogs = [];
  const runResponses = [];
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
    const url = new URL(request.url());
    if (url.origin === new URL(testInfo.project.use.baseURL).origin && url.pathname.startsWith('/data/')) {
      localDataRequests.push(url.href);
    }
  });

  // No interception, alternate data base, or fixture helper: exercise actual GitHub Raw/CORS.
  try {
    const indexResponsePending = page.waitForResponse(`${RAW_DATA_PREFIX}index.json`, { timeout: 60_000 });
    await page.goto('/');
    const indexResponse = await indexResponsePending;
    expect(indexResponse.ok(), 'Published index HTTP status').toBe(true);
    const index = await indexResponse.json();
    await testInfo.attach('published-index', { body: JSON.stringify(index, null, 2), contentType: 'application/json' });
    expect(Array.isArray(index.runFiles), 'Published index must list run files').toBe(true);
    expect(index.runFiles.length, 'Published index must be nonempty').toBeGreaterThan(0);

    const loadError = page.getByTestId('dashboard-data-error');
    const download = page.getByRole('button', { name: 'Download JSON', exact: true });
    // Wait for a terminal outcome, not an absent alert before fetch/adaptation settles.
    await expect.poll(async () => {
      if (await loadError.count()) return `load error: ${await loadError.innerText()}`;
      return await download.isEnabled() ? 'loaded' : 'loading';
    }, { timeout: 60_000, message: 'Published data must reach a terminal load state' }).not.toBe('loading');
    await expect(download, await loadError.count() ? await loadError.innerText() : 'Published data must load successfully').toBeEnabled();
    await expect(page.getByTestId('dashboard-data-loading')).toHaveCount(0);
    await expect(loadError).toHaveCount(0);
    await expect(page.getByTestId('fixture-data-warning')).toHaveCount(0);
    await expect(page.getByRole('alert')).toHaveCount(0);

    const downloadPending = page.waitForEvent('download');
    await download.click();
    const source = JSON.parse(await readFile(await (await downloadPending).path(), 'utf8'));
    expect(source.index).toEqual(index);
    expect(source.runs.length).toBe(index.runFiles.length);
    expect(source.runs.length).toBeGreaterThan(0);
    expect([...new Set(runResponses.map(({ url }) => url))].sort())
      .toEqual(index.runFiles.map((file) => `${RAW_DATA_PREFIX}${file}`).sort());
    expect(runResponses.every(({ ok }) => ok), 'All published run requests must succeed').toBe(true);
  } finally {
    expect.soft(pageErrors, 'Uncaught browser page errors').toEqual([]);
    expect.soft(dialogs, 'Unexpected JavaScript dialogs').toEqual([]);
    expect.soft(localDataRequests, 'Production must not substitute local data').toEqual([]);
    await expect.soft(page.getByTestId('fixture-data-warning')).toHaveCount(0);
    await expect.soft(page.getByRole('alert')).toHaveCount(0);
  }
});
