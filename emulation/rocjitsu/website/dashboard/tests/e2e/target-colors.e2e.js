import { expect, test } from '@playwright/test';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { installPublication, openDashboard } from './helpers/dashboard.js';

// gfx942 sorts alphabetically before gfx950 but first appears in a later run. Colors follow first appearance, so
// gfx1250 and gfx950 keep the blue and orange they had before gfx942 was published.
test('a target published later keeps earlier targets on their colors in the Overview metric strip', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  const publication = createMaintainedPublication();
  const latest = publication.runs.filter((run) => run.publication.source.branch === 'develop').at(-1);
  latest.benchmark_results.push(...latest.benchmark_results.filter(({ target }) => target === 'gfx950')
    .map((row) => ({ ...structuredClone(row), target: 'gfx942' })));
  await installPublication(page, { publication });
  await openDashboard(page, '/');
  const strip = page.getByTestId('metric-strip');
  const colorOf = (target) => strip.locator(`[data-configuration="${target}/default"]`).first().locator('svg').first()
    .evaluate((svg) => getComputedStyle(svg).color);
  expect(await colorOf('gfx1250')).toBe('rgb(15, 98, 254)');
  expect(await colorOf('gfx950')).toBe('rgb(194, 87, 12)');
  expect(await colorOf('gfx942')).toBe('rgb(21, 132, 99)');
});
