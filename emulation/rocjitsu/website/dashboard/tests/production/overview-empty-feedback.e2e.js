import { expect, test } from '@playwright/test';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
import { installPublication } from '../e2e/helpers/dashboard.js';
import { PUBLISHED_DATA_BASE_URL } from '../../scripts/dashboard-data-source.mjs';

test('empty production dashboard suppresses migration detail and makes all four icons gray', async ({ page }) => {
  await page.route('https://raw.githubusercontent.com/ROCm/rocm-systems/**', (route) => route.fulfill({ json: { schemaVersion: 1 } }));
  await page.goto('/');
  const banner = page.getByTestId('dashboard-data-error');
  await expect(banner).toContainText('No available test data');
  await expect(banner).not.toContainText('requires migration');
  await expect(banner.getByRole('button', { name: 'Retry' })).toBeVisible();
  for (const mode of ['light', 'dark']) {
    if (mode === 'dark') await page.getByRole('button', { name: 'Use dark theme' }).click();
    const cards = page.getByTestId('metric-strip').locator('[data-icon-tone]');
    await expect(cards).toHaveCount(4);
    const icons = await cards.evaluateAll((elements) => elements.map((element) => ({ tone: element.dataset.iconTone, color: getComputedStyle(element.querySelector('svg')).color })));
    expect(icons.map((icon) => icon.tone)).toEqual(['neutral', 'neutral', 'neutral', 'neutral']);
    expect(new Set(icons.map((icon) => icon.color)).size).toBe(1);
  }
});

for (const [factor, tone] of [[1, 'success'], [3, 'error']]) {
  test(`production Overview Perf change value and icon use ${tone} colors`, async ({ page }) => {
    const publication = createSchema2Publication();
    const latest = publication.runs.find((run) => run.id === 'fictional-develop-23');
    latest.configurations.forEach((configuration) => configuration.results.forEach((result) => {
      if (result.status === 'completed') result.durationSeconds *= factor;
    }));
    await installPublication(page, { publication, prefix: PUBLISHED_DATA_BASE_URL });
    await page.goto('/');
    const card = page.getByTestId('metric-card-perf-change');
    await expect(card).toHaveAttribute('data-tone', tone);
    await expect(card).toHaveAttribute('data-icon-tone', tone);
    for (const mode of ['light', 'dark']) {
      if (mode === 'dark') await page.getByRole('button', { name: 'Use dark theme' }).click();
      const colors = await card.evaluate((element) => ({ value: getComputedStyle(element.querySelector('p')).color, icon: getComputedStyle(element.querySelector('svg')).color }));
      expect(colors.icon).toBe(colors.value);
      const [red, green, blue] = colors.value.match(/\d+/g).map(Number);
      expect(tone === 'error' ? red > green && red > blue : green > red && green > blue).toBe(true);
    }
  });
}
