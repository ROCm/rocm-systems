import { expect, test } from '@playwright/test';
import { openDashboard } from './helpers/dashboard.js';
import { readChart } from './helpers/chart.js';

for (const range of [null, '1W']) {
  test(`fresh benchmark defaults have three measured charts (${range ?? 'default ALL'}) and visible IDs`, async ({ page }) => {
    await openDashboard(page, `/?view=benchmarks${range ? `&range=${range}` : ''}`);
    const cards = page.getByTestId('benchmark-grid-card');
    await expect(cards).toHaveCount(3);
    for (const card of await cards.all()) {
      const values = await readChart(card.getByRole('img'), (chart) => chart.getOption().series.filter((series) => series.type === 'line').flatMap((series) => series.data.map((point) => point?.value).filter(Number.isFinite)));
      expect(values.length).toBeGreaterThan(0);
      const id = card.getByRole('heading').locator('xpath=following-sibling::*[1]');
      await expect(id).toBeVisible();
      expect((await id.innerText()).trim()).not.toBe('');
    }
    expect(new URL(page.url()).searchParams.get('range')).toBe(range ?? 'ALL');
    await page.getByRole('button', { name: 'Add benchmarks', exact: true }).click();
    await page.getByRole('combobox', { name: 'Benchmarks to graph' }).click();
    const option = page.getByRole('listbox').getByRole('option').first();
    await expect(option.locator('.MuiListItemText-secondary span').first()).toBeVisible();
    await page.keyboard.press('Escape');
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await page.getByRole('tab', { name: 'Overview', exact: true }).click();
    await expect(page.getByRole('group', { name: 'History timeframe', exact: true }).getByRole('button', { name: range ? 'Trailing 7 days' : 'All available history', exact: true })).toHaveAttribute('aria-pressed', 'true');
  });
}
