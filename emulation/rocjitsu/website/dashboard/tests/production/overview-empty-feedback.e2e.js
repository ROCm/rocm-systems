import { expect, test } from '@playwright/test';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { installPublication } from '../e2e/helpers/dashboard.js';
import { PUBLISHED_DATA_BASE_URL } from '../../scripts/dashboard-data-source.mjs';

test('empty production dashboard suppresses migration detail and makes all four icons gray', async ({ page }) => {
  const publication = createMaintainedPublication();
  publication.runs[0].schema_version = 99;
  await installPublication(page, { publication, prefix: PUBLISHED_DATA_BASE_URL });
  await page.goto('/');
  const banner = page.getByTestId('dashboard-data-error');
  await expect(banner).toContainText('No available test data');
  await expect(banner).not.toContainText('requires migration');
  await expect(banner.getByRole('button', { name: 'Retry' })).toBeVisible();
  for (const mode of ['light', 'dark']) {
    if (mode === 'dark') await page.getByRole('button', { name: 'Use dark theme' }).click();
    const cards = page.getByTestId('metric-strip').locator('[data-tone]');
    await expect(cards).toHaveCount(4);
    const icons = await cards.evaluateAll((elements) => elements.map((element) => ({ tone: element.dataset.tone, color: getComputedStyle(element.querySelector('svg')).color })));
    expect(icons.map((icon) => icon.tone)).toEqual(['neutral', 'neutral', 'neutral', 'neutral']);
    expect(new Set(icons.map((icon) => icon.color)).size).toBe(1);
  }
});
