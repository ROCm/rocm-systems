import { expect, test } from '@playwright/test';
import { openDashboard } from './helpers/dashboard.js';
import { clickLastCompletedChartPoint } from './helpers/chart.js';

const errorsFor = (page) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
};

test('feedback: grouped benchmark options stay inside a short phone viewport and remain selectable', async ({ page }) => {
  const errors = errorsFor(page);
  await page.setViewportSize({ width: 390, height: 600 });
  await openDashboard(page, '/?view=benchmarks');
  await page.getByRole('button', { name: 'Add benchmarks', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Add benchmarks', exact: true });
  const picker = dialog.getByRole('combobox', { name: 'Benchmarks to graph' });
  await picker.click();
  const list = page.getByRole('listbox');
  await expect(list).toBeVisible();
  const groups = await list.locator('.MuiAutocomplete-groupLabel').allTextContents();
  expect(groups).toEqual(['Llama', 'Triton']);
  const optionCount = await list.getByRole('option').count();
  expect(optionCount).toBeGreaterThan(4);
  const contained = async () => {
    const box = await list.boundingBox();
    return Boolean(box && box.y >= 0 && box.y + box.height <= 600 && box.x >= 0 && box.x + box.width <= 390);
  };
  await expect.poll(contained).toBe(true);
  expect(await list.evaluate((element) => getComputedStyle(element).overflowY)).toBe('auto');
  const choice = list.getByRole('option', { name: /^Fictional added workload E\b/ });
  await choice.scrollIntoViewIfNeeded();
  await choice.click();
  await expect(choice).toHaveAttribute('aria-selected', 'true');
  await expect(list.getByRole('option')).toHaveCount(optionCount);
  await expect.poll(contained).toBe(true);
  await picker.press('Escape');
  await dialog.getByRole('button', { name: 'Apply selection' }).click();
  await expect(page.getByTestId('benchmark-grid-card').getByRole('heading', { name: 'Fictional added workload E', exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('feedback: direct chart details remain standalone without stale-model pointer errors', async ({ page }) => {
  const errors = errorsFor(page);
  await openDashboard(page, '/?view=benchmarks');
  const card = page.getByTestId('benchmark-grid-card').filter({ has: page.getByRole('heading', { name: 'Fictional GEMM', exact: true }) });
  await clickLastCompletedChartPoint(card.getByRole('img'));
  const details = page.getByRole('dialog').filter({ has: page.getByRole('button', { name: 'Close details', exact: true }) });
  await expect(details).toBeVisible();
  await expect(page.locator('[role="dialog"][aria-labelledby="inspect-results-title"]')).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(details).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('feedback: nested result details preserve inspector choice, focus and repeated drill-down', async ({ page }) => {
  const errors = errorsFor(page);
  await openDashboard(page, '/?view=benchmarks');
  const inspect = page.getByRole('button', { name: 'Inspect Fictional GEMM results', exact: true });
  await inspect.click();
  const inspector = page.locator('[role="dialog"][aria-labelledby="inspect-results-title"]');
  const picker = inspector.getByRole('combobox');
  const open = inspector.getByRole('button', { name: 'Open result details', exact: true });
  const details = page.getByRole('dialog').filter({ has: page.getByRole('button', { name: 'Close details', exact: true }) });
  await picker.fill('fictional-develop-20');
  await page.getByRole('option').first().click();
  const selected = await picker.inputValue();
  await open.click();
  await expect(details).toBeVisible();
  await expect(inspector).toHaveCount(1);
  await expect.poll(() => page.evaluate(() => Boolean(document.activeElement?.closest('[role="dialog"]')?.querySelector('[aria-label="Close details"]')))).toBe(true);
  await page.keyboard.press('Escape');
  await expect(details).toHaveCount(0);
  await expect(inspector).toBeVisible();
  await expect(picker).toHaveValue(selected);
  await expect(open).toBeFocused();
  await picker.fill('fictional-develop-19');
  await page.getByRole('option').first().click();
  const another = await picker.inputValue();
  expect(another).not.toBe(selected);
  await open.click();
  await expect(details).toBeVisible();
  await details.getByRole('button', { name: 'Close details', exact: true }).click();
  await expect(details).toHaveCount(0);
  await expect(inspector).toBeVisible();
  await expect(picker).toHaveValue(another);
  await expect(open).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(inspector).toHaveCount(0);
  expect(errors).toEqual([]);
});
