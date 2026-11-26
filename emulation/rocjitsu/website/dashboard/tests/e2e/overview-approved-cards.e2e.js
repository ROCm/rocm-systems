import { expect, test } from '@playwright/test';
import { expectNoDocumentOverflow, openDashboard } from './helpers/dashboard.js';

test('approved cards retain single-configuration, empty-selection and unavailable states', async ({ page }) => {
  await openDashboard(page, '/?view=overview&targets=gfx950&modes=single&suites=Triton');
  const cards = page.locator('[data-testid^="metric-card-"]');
  await expect(cards).toHaveCount(4);
  await expect(cards.first().locator('[data-configuration]')).toHaveCount(1);
  await openDashboard(page, '/?view=overview&targets=&modes=&suites=');
  await expect(cards).toHaveCount(4);
  await expect(cards.getByText('No configurations selected', { exact: true })).toHaveCount(4);
  await openDashboard(page, '/?view=overview&targets=gfx950&modes=single&suites=');
  await expect(cards.getByText('No shared workloads', { exact: true })).toHaveCount(4);
});

for (const width of [1440, 390]) {
  test(`approved Overview cards use compact neutral rows at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    await openDashboard(page, '/?view=overview');
    const cards = page.locator('[data-testid^="metric-card-"]');
    await expect(cards).toHaveCount(4);
    await expect(cards.filter({ has: page.locator('h3') })).toHaveCount(4);
    expect(await cards.evaluateAll((nodes) => nodes.every((node) => node.classList.contains('MuiPaper-outlined')))).toBe(true);
    if (width === 1440) {
      const geometry = await cards.evaluateAll((nodes) => nodes.map((node) => ({ top: node.getBoundingClientRect().top, height: node.getBoundingClientRect().height, footer: node.lastElementChild.getBoundingClientRect().top })));
      expect(new Set(geometry.map((item) => item.top)).size).toBe(1);
      expect(new Set(geometry.map((item) => item.height)).size).toBe(1);
      expect(new Set(geometry.map((item) => item.footer)).size).toBe(1);
    }
    await expectNoDocumentOverflow(page);
    await page.getByRole('button', { name: 'Use dark theme' }).click();
    await expect(cards).toHaveCount(4);
    await expectNoDocumentOverflow(page);
  });
}
