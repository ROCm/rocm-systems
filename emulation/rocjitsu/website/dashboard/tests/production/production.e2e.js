import { readdir, readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
import { downloadSource, installPublication, openDashboard, RAW_DATA_PREFIX, ready } from '../e2e/helpers/dashboard.js';

async function artifactFiles(directory, prefix = '') {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const relative = `${prefix}${entry.name}`;
    if (entry.isDirectory()) files.push(...await artifactFiles(new URL(`${entry.name}/`, directory), `${relative}/`));
    else files.push({ relative, url: new URL(entry.name, directory) });
  }
  return files;
}

test('production artifact contains application assets but no fictional publication or local data directory', async ({ request }) => {
  const files = await artifactFiles(new URL('../../dist/', import.meta.url));
  expect(files.some(({ relative }) => relative === 'index.html')).toBe(true);
  expect(files.some(({ relative }) => relative.startsWith('assets/') && relative.endsWith('.js'))).toBe(true);
  expect(files.filter(({ relative }) => /(^|\/)(data|fixtures|tests)\//.test(relative) || relative.endsWith('.json'))).toEqual([]);
  for (const file of files.filter(({ relative }) => /\.(html|js|map)$/.test(relative))) {
    expect(await readFile(file.url, 'utf8'), file.relative)
      .not.toMatch(/fictional-develop-\d|fictional-branch-\d|Fictional GEMM|fictional-current\.json/);
  }
  // A Vite SPA fallback may return HTML with 200; it must never return fixture JSON.
  const response = await request.get('/data/metadata.json');
  expect(response.headers()['content-type'] ?? '').not.toMatch(/json/i);
  expect(await response.text()).not.toContain('fictional');
});

test('production consumer reads deterministic intercepted schema 2 at the published origin without fixture fallback', async ({ page }) => {
  const localDataRequests = [];
  page.on('request', (request) => {
    const url = new URL(request.url());
    if (url.origin === new URL(page.url()).origin && url.pathname.startsWith('/data/')) localDataRequests.push(url.href);
  });
  const { publication, requests } = await installPublication(page, { prefix: RAW_DATA_PREFIX });
  await openDashboard(page, '/?view=compare&compareCandidate=fictional-branch-01&compareBaseline=fictional-develop-20&targets=gfx950&modes=MT');
  await expect(page.getByTestId('fixture-data-warning')).toHaveCount(0);
  await expect(page.getByTestId('candidate-run-selected-identity')).toContainText('fictional-branch-01');
  await expect(page.getByTestId('baseline-run-selected-identity')).toContainText('fictional-develop-20');
  await expect(page.getByRole('img', { name: 'Performance change by benchmark comparison chart' })).toBeVisible();
  expect(await downloadSource(page)).toEqual(publication);
  expect(requests).toContain(`${RAW_DATA_PREFIX}metadata.json`);
  expect(requests).toContain(`${RAW_DATA_PREFIX}index.json`);
  expect(requests.filter((url) => url.startsWith(`${RAW_DATA_PREFIX}runs/`))).toHaveLength(publication.runs.length);
  expect(requests.some((url) => url.startsWith(`${RAW_DATA_PREFIX}test-catalogs/`))).toBe(true);
  expect(localDataRequests).toEqual([]);
  // This is an origin/consumer boundary test, not evidence of a live schema-2
  // publisher, live GitHub CORS headers, or Chromium's HTTP cache hit rate.
  await page.getByRole('button', { name: 'Reload all data' }).click();
  await ready(page);
  const generation = await page.evaluate(() => localStorage.getItem('rocjitsu-data-cache-generation'));
  expect(generation).toBeTruthy();
  const refreshed = requests.filter((url) => new URL(url).searchParams.has('reload'));
  expect(refreshed).toHaveLength(publication.runs.length + Object.keys(publication.catalogs).length + 2);
  expect(refreshed.every((url) => new URL(url).searchParams.get('reload') === generation)).toBe(true);
  expect(await downloadSource(page)).toEqual(publication);
});

test('production rejects schema 1 with generic no-data UI and no fixture substitution', async ({ page }) => {
  const publication = createSchema2Publication();
  publication.metadata.schemaVersion = 1;
  const { requests } = await installPublication(page, { prefix: RAW_DATA_PREFIX, publication });
  await page.goto('/');
  await expect(page.getByTestId('dashboard-data-error')).toContainText('No available test data');
  await expect(page.getByTestId('dashboard-data-error')).not.toContainText('requires migration');
  await expect(page.getByTestId('dashboard-data-error').getByRole('button', { name: 'Retry', exact: true })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Reload all data' })).toBeEnabled();
  await expect(page.getByTestId('fixture-data-warning')).toHaveCount(0);
  await expect(page.getByRole('img')).toHaveCount(0);
  // The document preloads index.json; a no-store consumer fetch may request it
  // again. Assert the resource boundary, not an incidental preload hit count.
  expect([...new Set(requests)].sort()).toEqual([`${RAW_DATA_PREFIX}index.json`, `${RAW_DATA_PREFIX}metadata.json`]);
  await page.getByRole('tab', { name: 'Branch Runs', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'No published branch runs' })).toBeVisible();
  await expect(page.getByTestId('dashboard-data-error')).toBeVisible();
  await expect(page.getByTestId('branch-config-gfx1250-ST')).toHaveCount(0);
});
