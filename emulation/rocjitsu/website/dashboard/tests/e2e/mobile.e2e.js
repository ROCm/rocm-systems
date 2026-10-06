import { expect, test } from '@playwright/test';
import { expectDialogTypographyContained } from './helpers/dialog.js';

test('does not overflow the mobile viewport', async ({ page }) => {
  await page.goto('/');
  const dimensions = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    document: document.documentElement.scrollWidth,
  }));
  expect(dimensions.document).toBeLessThanOrEqual(dimensions.viewport);
  await expect(page.getByRole('heading', { name: 'Rocjitsu Simulation Performance' })).toBeVisible();
});

test('replaces the recent-runs table with run cards that keep their labels', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByTestId('recent-runs-table')).toHaveCount(0);
  const newestExecution = page.getByTestId('mobile-run').first();
  await expect(newestExecution).toContainText('8418072e');
  await expect(newestExecution).toContainText('Most recent');
  await expect(newestExecution).toContainText('Historical rerun');
  await expect(page.getByTestId('mobile-run').filter({ hasText: '31369c4d' })).toContainText('Latest commit');
});

test('keeps phone-only controls, badges, and dialog text within their containers', async ({ page }) => {
  await page.goto('/');

  const navigation = page.getByTestId('dashboard-navigation');
  const failuresTab = page.getByRole('tab', { name: /Failures/ });
  const badge = failuresTab.locator('.MuiBadge-badge');
  const [navigationBox, tabBox, badgeBox] = await Promise.all([
    navigation.boundingBox(),
    failuresTab.boundingBox(),
    badge.boundingBox(),
  ]);
  expect(badgeBox.y).toBeGreaterThanOrEqual(navigationBox.y);
  expect(badgeBox.y + badgeBox.height).toBeLessThanOrEqual(navigationBox.y + navigationBox.height);
  expect(badgeBox.x).toBeGreaterThanOrEqual(tabBox.x);
  expect(badgeBox.x + badgeBox.width).toBeLessThanOrEqual(tabBox.x + tabBox.width);

  await page.getByTestId('latest-results')
    .getByRole('button', { name: /^Open GEMM FP16 1024³ result details/ })
    .first()
    .click();
  const dialog = page.getByRole('dialog');
  await expectDialogTypographyContained(dialog);
  await page.getByRole('button', { name: 'Close details' }).click();

  await page.getByRole('tab', { name: 'Benchmarks' }).click();
  const benchmarkInputBox = await page.getByRole('combobox', { name: 'Benchmark' })
    .locator('..')
    .boundingBox();
  const detailsButtonBox = await page.getByRole('button', { name: 'Disable details on click' })
    .boundingBox();
  expect(detailsButtonBox.y).toBeGreaterThanOrEqual(benchmarkInputBox.y + benchmarkInputBox.height);
});

test('keeps every dashboard view reachable on a phone', async ({ page }) => {
  await page.goto('/');

  for (const [tab, heading] of [
    ['Benchmarks', 'Benchmark Explorer'],
    ['Run Comparison', 'Performance Change by Benchmark'],
    ['Plugin Comparison', 'Plugin Comparison'],
  ]) {
    await page.getByRole('tab', { name: tab }).click();
    await expect(page.getByText(heading).first()).toBeVisible();
    const widths = await page.evaluate(() => ({
      viewport: document.documentElement.clientWidth,
      document: document.documentElement.scrollWidth,
    }));
    expect(widths.document).toBeLessThanOrEqual(widths.viewport);
  }

  await page.getByRole('tab', { name: /Failures/ }).click();
  await expect(page.getByText('Run Reliability')).toBeVisible();
});

test('mobile threading switches scoped data without losing the active tab', async ({ page }) => {
  const { serveThreadingDataset, switchThreading } = await import('./helpers/threading.js');
  await serveThreadingDataset(page);
  await page.goto('/');
  await switchThreading(page, 'Single-thread');
  await expect(page.getByTestId('targets-filter')).toContainText('gfx950');
  for (const tab of ['Benchmarks', 'Run Comparison', 'Plugin Comparison', 'Failures', 'Overview']) {
    const navigation = page.getByRole('tab', { name: tab, exact: true });
    await navigation.scrollIntoViewIfNeeded();
    await navigation.click();
    await switchThreading(page, 'Default');
    await expect(navigation).toHaveAttribute('aria-selected', 'true');
    await switchThreading(page, 'Single-thread');
    await expect(navigation).toHaveAttribute('aria-selected', 'true');
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
