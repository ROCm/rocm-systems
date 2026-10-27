import { expect, test } from '@playwright/test';
import { openDashboard } from './helpers/dashboard.js';

async function expectRunStyle(page, picker) {
  await picker.click();
  const option = page.getByRole('listbox').getByRole('option').first();
  await expect(option).toContainText('Test run ·');
  await expect(option).toContainText('Commit ·');
  await expect(option.locator('.MuiListItemText-primary')).toBeVisible();
  await page.keyboard.press('Escape');
}

test('all run pickers share the comparison option layout and retain exact run search', async ({ page }) => {
  await openDashboard(page, '/?view=branch&detail=1');
  for (const label of ['Candidate', 'Reference']) await expectRunStyle(page, page.getByRole('combobox', { name: label, exact: true }));
  const reference = page.getByRole('combobox', { name: 'Reference', exact: true });
  await reference.fill('fictional-develop-20');
  await page.getByRole('option').first().click();
  await expect(page.getByTestId('reference-selected-identity')).toContainText('fictional-develop-20');
  await openDashboard(page, '/?view=compare');
  for (const label of ['Candidate run', 'Baseline run']) await expectRunStyle(page, page.getByRole('combobox', { name: label, exact: true }));
  await openDashboard(page, '/?view=benchmarks');
  await page.getByRole('button', { name: 'Inspect Fictional GEMM results', exact: true }).click();
  await expectRunStyle(page, page.getByRole('dialog').getByRole('combobox'));
});

test('pull request filter uses the styled anchored menu and keeps filtering behavior', async ({ page }) => {
  for (const width of [1440, 390]) {
  await page.setViewportSize({ width, height: 900 });
  await openDashboard(page, '/?view=branch');
  const picker = page.getByRole('combobox', { name: 'Pull request', exact: true });
  const fieldWidth = await picker.evaluate((node) => node.closest('.MuiInputBase-root').getBoundingClientRect().width);
  expect(await picker.evaluate((node) => node.tagName)).not.toBe('SELECT');
  await picker.click();
  await expect(page.getByRole('listbox')).toBeVisible();
  await expect.poll(() => page.getByRole('listbox').evaluate((node) => node.closest('.MuiPaper-root').getBoundingClientRect().width)).toBeCloseTo(fieldWidth, 0);
  await page.getByRole('option', { name: 'With PR', exact: true }).click();
  await expect(picker).toHaveText('With PR');
  await picker.click();
  await page.getByRole('option', { name: 'All branches', exact: true }).click();
  await expect(picker).toHaveText('All branches');
  }
});

test('Add benchmarks popup uses density-matched text, icons and row spacing', async ({ page }) => {
  await openDashboard(page, '/?view=benchmarks');
  await page.getByRole('button', { name: 'Add benchmarks', exact: true }).click();
  await page.getByRole('combobox', { name: 'Benchmarks to graph' }).click();
  const option = page.getByRole('listbox').getByRole('option').first();
  const expectedSize = await page.locator('body').evaluate((body) => 20 * Number(getComputedStyle(body).zoom));
  await expect.poll(() => option.locator('.MuiListItemText-primary').evaluate((node) => parseFloat(getComputedStyle(node).lineHeight)))
    .toBeCloseTo(expectedSize, 3);
  await expect.poll(() => option.locator('svg').evaluate((node) => parseFloat(getComputedStyle(node).fontSize)))
    .toBeCloseTo(expectedSize, 3);
});
