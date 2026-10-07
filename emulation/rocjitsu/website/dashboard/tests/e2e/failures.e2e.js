import { expect, test } from '@playwright/test';

test('failures shows reliability and a bounded, newest-first case list', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: /Failures/ }).click();

  await expect(page.getByText('Run Reliability')).toBeVisible();
  await expect(page.getByText('Coverage for the latest 20 official runs under the current filters.')).toBeVisible();
  await expect(page.getByText(/^\d{1,2}\/20$/)).toBeVisible();
  await expect(page.getByText('Runs below 100%')).toBeVisible();
  await expect(page.getByRole('img', { name: 'Run reliability coverage trend' })).toBeVisible();

  const progressBars = page.getByTestId('reliability-progress');
  const barPositions = await progressBars.evaluateAll((elements) => elements.map((element) => element.getBoundingClientRect().x));
  expect(new Set(barPositions.map((position) => Math.round(position))).size).toBe(1);

  const cases = page.getByTestId('failure-cases');
  await expect(cases.getByText('Failed and Timed-Out Cases')).toBeVisible();
  const renderedCases = cases.getByTestId('failure-case');
  await expect(cases.getByTestId('failure-total')).toHaveText('5');
  await expect(renderedCases).toHaveCount(5);
  await expect(cases.getByTestId('failure-range')).toHaveText('1–5 of 5 cases');

  const timestamps = await renderedCases.locator('[data-failure-time]').evaluateAll(
    (elements) => elements.map((element) => element.dataset.failureTime),
  );
  expect(timestamps).toEqual([...timestamps].sort().reverse());
});
