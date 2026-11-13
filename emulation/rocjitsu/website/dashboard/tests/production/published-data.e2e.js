import { test } from '@playwright/test';
import { checkPublishedData } from './helpers/published-data.js';

test('new production build handles the current published state without browser errors', async ({ page }, testInfo) => {
  test.skip(process.env.PLAYWRIGHT_PUBLISHED_DATA !== '1', 'Opt in to live network coverage with PLAYWRIGHT_PUBLISHED_DATA=1');
  test.setTimeout(90_000);
  // No interception, alternate data base, or fixture helper: exercise actual GitHub Raw/CORS.
  await checkPublishedData(page, testInfo);
});
