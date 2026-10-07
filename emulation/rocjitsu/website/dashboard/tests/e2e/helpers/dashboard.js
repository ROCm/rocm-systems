import { readFile } from 'node:fs/promises';
import { expect } from '@playwright/test';
import { createSchema2Publication } from '../../fixtures/schema2Dataset.js';

export const RAW_DATA_PREFIX = 'https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/';

export async function ready(page) {
  await expect(page.getByRole('button', { name: 'Download JSON', exact: true })).toBeEnabled();
  await expect(page.getByTestId('dashboard-data-loading')).toHaveCount(0);
  await expect(page.getByTestId('dashboard-data-error')).toHaveCount(0);
}

export async function openDashboard(page, url = '/') {
  await page.goto(url);
  await ready(page);
}

export async function selectRun(page, label, runId) {
  const picker = page.getByRole('combobox', { name: label, exact: true });
  await picker.fill(runId);
  const choice = page.getByRole('option').filter({ hasText: runId });
  await expect(choice).toHaveCount(1);
  await picker.press('ArrowDown');
  await picker.press('Enter');
}

export async function downloadSource(page) {
  const pending = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download JSON', exact: true }).click();
  const download = await pending;
  expect(download.suggestedFilename()).toBe('rocjitsu-simulation-benchmark-data.json');
  return JSON.parse(await readFile(await download.path(), 'utf8'));
}

// All interception is test-only, at the real consumer's HTTP boundary. No live
// publisher or timing measurements are implied. Keep handlers until teardown.
export async function installPublication(page, {
  publication = createSchema2Publication(), prefix = '/data/', beforeResponse,
} = {}) {
  const files = {
    'metadata.json': publication.metadata, 'index.json': publication.index,
    ...publication.catalogs,
    ...Object.fromEntries(publication.runs.map((run) => [`runs/${run.id}.json`, run])),
  };
  const requests = [];
  await page.route(prefix.startsWith('https:') ? `${prefix}**` : '**/data/**', async (route) => {
    const url = new URL(route.request().url());
    const relative = prefix.startsWith('https:')
      ? url.href.slice(prefix.length).split('?')[0]
      : url.pathname.slice(prefix.length);
    requests.push(url.href);
    const override = await beforeResponse?.({ route, relative, url });
    const response = override ?? (Object.hasOwn(files, relative)
      ? { status: 200, contentType: 'application/json', json: files[relative] }
      : { status: 404, contentType: 'text/plain', body: 'No fictional resource' });
    await route.fulfill({ ...response, headers: { 'access-control-allow-origin': '*', ...response.headers } });
  });
  return { publication, requests };
}

// Routing disables Chromium's HTTP cache. Assert the consumer's fetch policy and
// generation contract, NOT a claim that intercepted responses came from disk.
export async function recordFetchPolicies(page) {
  await page.addInitScript(() => {
    const original = window.fetch.bind(window);
    window.__dashboardFetches = [];
    window.fetch = (input, init) => {
      window.__dashboardFetches.push({ url: String(input), cache: init?.cache ?? 'default' });
      return original(input, init);
    };
  });
}

export async function fetchPolicies(page) {
  return page.evaluate(() => window.__dashboardFetches.filter(({ url }) => url.includes('/data/')));
}

export async function ensureGridBenchmark(page, name) {
  await page.getByRole('button', { name: 'Add benchmarks', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Add benchmarks', exact: true });
  const picker = dialog.getByRole('combobox', { name: 'Benchmarks to graph' });
  await picker.fill(name);
  const option = page.getByRole('option').filter({ hasText: name });
  await expect(option).toHaveCount(1);
  if (await option.getAttribute('aria-selected') !== 'true') await option.click();
  await picker.press('Escape');
  await dialog.getByRole('button', { name: 'Apply selection' }).click();
}

export async function inspectResult(page, name, runId, target = 'gfx1250', mode = 'ST') {
  await page.getByRole('button', { name: `Inspect ${name} results`, exact: true }).click();
  const dialog = page.getByRole('dialog');
  const picker = dialog.getByRole('combobox', { name: `${name} result`, exact: true });
  await picker.fill(runId);
  const option = page.getByRole('option').filter({ hasText: runId }).filter({ hasText: `${target} ${mode}` });
  await expect(option).toHaveCount(1);
  await option.click();
  await dialog.getByRole('button', { name: 'Open result details' }).press('Enter');
  return page.getByRole('dialog').filter({ has: page.getByRole('button', { name: 'Close details', exact: true }) });
}

export async function expectNoDocumentOverflow(page) {
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth
    - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
}

export async function expectLocalScroll(region, { horizontal = false } = {}) {
  await region.scrollIntoViewIfNeeded();
  const before = await region.evaluate((element, sideways) => ({
    pageY: window.scrollY, pageX: window.scrollX,
    range: sideways ? element.scrollWidth - element.clientWidth : element.scrollHeight - element.clientHeight,
  }), horizontal);
  expect(before.range).toBeGreaterThan(0);
  await region.evaluate((element, sideways) => {
    if (sideways) element.scrollLeft = element.scrollWidth;
    else element.scrollTop = element.scrollHeight;
  }, horizontal);
  await expect.poll(() => region.evaluate((element, sideways) => sideways
    ? element.scrollLeft : element.scrollTop, horizontal)).toBeGreaterThan(0);
  expect(await region.evaluate(() => ({ pageY: window.scrollY, pageX: window.scrollX })))
    .toEqual({ pageY: before.pageY, pageX: before.pageX });
}
