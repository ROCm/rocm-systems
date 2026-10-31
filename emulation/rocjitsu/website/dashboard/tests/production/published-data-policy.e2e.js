import { expect, test } from '@playwright/test';
import { installPublication } from '../e2e/helpers/dashboard.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { checkPublishedData, RAW_DATA_PREFIX } from './helpers/published-data.js';

// Deterministic boundary controls, separate from the un-intercepted live smoke.
test('published smoke accepts index 404 only with the real empty dashboard', async ({ page }, testInfo) => {
  await installPublication(page, {
    prefix: RAW_DATA_PREFIX,
    beforeResponse: () => ({ status: 404, contentType: 'text/plain', body: 'Not found' }),
  });
  await checkPublishedData(page, testInfo);
});

test('published smoke rejects an arbitrary error banner even for index 404', async ({ page }, testInfo) => {
  await page.addInitScript(() => {
    const observer = new MutationObserver(() => {
      const banner = document.querySelector('[data-testid="dashboard-data-error"]');
      if (banner) {
        observer.disconnect();
        banner.textContent = 'An unrelated application error';
      }
    });
    observer.observe(document, { childList: true, subtree: true });
  });
  await installPublication(page, {
    prefix: RAW_DATA_PREFIX,
    beforeResponse: () => ({ status: 404, body: 'Not found' }),
  });
  await expect(checkPublishedData(page, testInfo)).rejects.toThrow(/toHaveText/);
});

test('published smoke retains nonempty load, download, and run request checks', async ({ page }, testInfo) => {
  await installPublication(page, { prefix: RAW_DATA_PREFIX });
  await checkPublishedData(page, testInfo);
});

test('published smoke accepts valid empty index with a successful empty source download', async ({ page }, testInfo) => {
  await installPublication(page, {
    prefix: RAW_DATA_PREFIX,
    publication: { index: { runFiles: [] }, runs: [] },
  });
  await checkPublishedData(page, testInfo);
});

for (const [name, response, failure] of [
  ['missing runFiles', { json: {} }, /Published index must list run files/],
  ['null index', { json: null }, /Published index must list run files/],
  ['invalid JSON', { body: '{', contentType: 'application/json' }, /JSON/],
  ['HTML instead of index', { body: '<html>Not data</html>', contentType: 'text/html' }, /JSON/],
  ['HTTP 500', { status: 500, body: 'Unavailable' }, /Published index HTTP status/],
  ['HTTP 403', { status: 403, body: 'Forbidden' }, /Published index HTTP status/],
]) {
  test(`published smoke rejects ${name}`, async ({ page }, testInfo) => {
    await installPublication(page, {
      prefix: RAW_DATA_PREFIX,
      beforeResponse: () => ({ status: 200, contentType: 'application/json', ...response }),
    });
    await expect(checkPublishedData(page, testInfo)).rejects.toThrow(failure);
  });
}

test('published smoke rejects network failure', async ({ page }, testInfo) => {
  await page.route(`${RAW_DATA_PREFIX}**`, (route) => route.abort('failed'));
  await expect(checkPublishedData(page, testInfo, { timeout: 2_000 })).rejects.toThrow(/waitForResponse/);
});

for (const badRun of ['HTTP 404', 'invalid schema']) {
  test(`published smoke rejects run ${badRun}`, async ({ page }, testInfo) => {
    const publication = createMaintainedPublication();
    publication.runs = publication.runs.slice(0, 1);
    publication.index.runFiles = publication.index.runFiles.slice(0, 1);
    if (badRun === 'invalid schema') publication.runs[0].schema_version = 99;
    await installPublication(page, {
      prefix: RAW_DATA_PREFIX,
      publication,
      beforeResponse: ({ relative }) => relative.startsWith('runs/') && badRun === 'HTTP 404'
        ? { status: 404, body: 'Missing run' } : undefined,
    });
    await expect(checkPublishedData(page, testInfo)).rejects.toThrow(/Published data must load successfully|No available test data/);
  });
}
