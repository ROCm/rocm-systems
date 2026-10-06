import { expect, test } from '@playwright/test';

const RAW_DATA_PREFIX = (
  'https://raw.githubusercontent.com/ROCm/rocm-systems/'
  + 'refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/'
);

function isRunUrl(url) {
  return url.startsWith(`${RAW_DATA_PREFIX}runs/`);
}

test('production loads GitHub Raw data and reuses refreshed files from browser cache', async ({
  context,
  page,
}) => {
  const cdp = await context.newCDPSession(page);
  const requestUrls = new Map();
  const requestedUrls = [];
  const cachedRequestIds = new Set();
  let initialIndexResponse = null;

  cdp.on('Network.requestWillBeSent', ({ requestId, request }) => {
    requestUrls.set(requestId, request.url);
    requestedUrls.push(request.url);
  });
  cdp.on('Network.requestServedFromCache', ({ requestId }) => {
    cachedRequestIds.add(requestId);
  });
  cdp.on('Network.responseReceived', ({ requestId, response }) => {
    if (response.fromDiskCache) cachedRequestIds.add(requestId);
  });
  await cdp.send('Network.enable');

  page.on('response', (response) => {
    if (response.url() === `${RAW_DATA_PREFIX}index.json`) {
      initialIndexResponse = response;
    }
  });

  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeEnabled();

  expect(initialIndexResponse).not.toBeNull();
  expect(await initialIndexResponse.headerValue('access-control-allow-origin')).toBe('*');
  expect(await initialIndexResponse.headerValue('content-type')).toMatch(/json|text\/plain/i);
  expect(requestedUrls).toContain(`${RAW_DATA_PREFIX}metadata.json`);
  expect(requestedUrls).toContain(`${RAW_DATA_PREFIX}index.json`);
  expect(requestedUrls.some(isRunUrl)).toBe(true);
  expect(requestedUrls.some((url) => url.startsWith(`${RAW_DATA_PREFIX}test-catalogs/`)))
    .toBe(true);

  cachedRequestIds.clear();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeEnabled();
  await expect.poll(() => [...cachedRequestIds]
    .map((requestId) => requestUrls.get(requestId))
    .some((url) => url && isRunUrl(url))).toBe(true);

  const reloadStart = requestedUrls.length;
  await page.getByRole('button', { name: 'Reload all data' }).click();
  await expect.poll(() => requestedUrls.slice(reloadStart).some((url) => (
    url.startsWith(RAW_DATA_PREFIX) && new URL(url).searchParams.has('reload')
  ))).toBe(true);
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeEnabled();

  const savedGeneration = await page.evaluate(
    () => window.localStorage.getItem('rocjitsu-data-cache-generation'),
  );
  expect(savedGeneration).toBeTruthy();

  const refreshedUrls = requestedUrls.slice(reloadStart)
    .filter((url) => url.startsWith(RAW_DATA_PREFIX));
  expect(refreshedUrls.some((url) => url.includes('/metadata.json?'))).toBe(true);
  expect(refreshedUrls.some((url) => url.includes('/index.json?'))).toBe(true);
  expect(refreshedUrls.some(isRunUrl)).toBe(true);
  expect(refreshedUrls.every(
    (url) => new URL(url).searchParams.get('reload') === savedGeneration,
  )).toBe(true);

  cachedRequestIds.clear();
  const refreshStart = requestedUrls.length;
  await page.reload();
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeEnabled();

  const normalRefreshRuns = requestedUrls.slice(refreshStart).filter(isRunUrl);
  expect(normalRefreshRuns.length).toBeGreaterThan(0);
  expect(normalRefreshRuns.every(
    (url) => new URL(url).searchParams.get('reload') === savedGeneration,
  )).toBe(true);
  await expect.poll(() => [...cachedRequestIds]
    .map((requestId) => requestUrls.get(requestId))
    .some((url) => url && isRunUrl(url)
      && new URL(url).searchParams.get('reload') === savedGeneration)).toBe(true);
});
