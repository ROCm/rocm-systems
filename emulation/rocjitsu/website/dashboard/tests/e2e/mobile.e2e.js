import { expect, test } from '@playwright/test';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
import { readChart } from './helpers/chart.js';
import { expectDialogTypographyContained } from './helpers/dialog.js';
import { ensureGridBenchmark, expectLocalScroll, expectNoDocumentOverflow, inspectResult, openDashboard } from './helpers/dashboard.js';

test('phone filters, all-page navigation, trend tap and result details stay usable in both themes', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  await openDashboard(page, '/?compareCandidate=fictional-develop-23&compareBaseline=fictional-develop-20');
  const filters = page.getByRole('button', { name: /^Filters ·/ });
  await expect(filters).toHaveAttribute('aria-expanded', 'false');
  await expect(page.getByRole('checkbox', { name: 'MT', exact: true })).toHaveCount(0);
  await filters.tap();
  await expect(filters).toHaveAttribute('aria-expanded', 'true');
  await page.getByRole('checkbox', { name: 'MT', exact: true }).uncheck();
  await page.getByRole('checkbox', { name: 'gfx950', exact: true }).check();
  await filters.tap();
  await expect(filters).toContainText('2 targets / 2 suites / ST');
  await expect(filters).toHaveAttribute('aria-expanded', 'false');
  await expectNoDocumentOverflow(page);
  const recent = page.getByRole('region', { name: 'Recent runs scroll area' });
  await expectLocalScroll(recent, { horizontal: true });
  await expectLocalScroll(recent);
  await expect(page.getByTestId('recent-runs').getByRole('status')).toHaveText('Showing 1–20 of 24 runs');

  // Tap with the phone's touch input, then explicitly leave: touch inspection must
  // not be cleared like mouse hover when the finger lifts.
  const trend = page.getByRole('group', { name: 'Inspect performance trend' });
  await trend.scrollIntoViewIfNeeded();
  const pixel = await readChart(trend.getByRole('img'), (instance) => {
    const option = instance.getOption();
    const point = option.series[0].data.findLast((entry) => Number.isFinite(entry[1]));
    return instance.convertToPixel({ seriesIndex: 0 }, point);
  });
  await trend.tap({ position: { x: pixel[0], y: pixel[1] } });
  await trend.dispatchEvent('pointerleave', { pointerType: 'touch' });
  await expect(page.getByTestId('performance-trend').getByRole('status')).toContainText(createSchema2Publication().runs.find((run) => run.id === 'fictional-develop-23').source.commit.slice(0, 8));
  await trend.press('Escape');
  await expect(page.getByTestId('performance-trend').getByRole('status')).toHaveCount(0);

  for (const theme of ['light', 'dark']) {
    if (theme === 'dark') await page.getByRole('button', { name: 'Use dark theme' }).tap();
    for (const name of ['Branch Runs', 'Benchmarks', 'Run Comparison', 'Overview']) {
      const tab = page.getByRole('tab', { name, exact: true });
      await tab.tap();
      await expect(tab).toHaveAttribute('aria-selected', 'true');
      await expect(page.getByRole('heading', { name, level: 1, exact: true })).toBeVisible();
      await expectNoDocumentOverflow(page);
    }
  }
  await filters.tap();
  await expect(page.getByRole('checkbox', { name: 'MT', exact: true })).not.toBeChecked();
  await filters.tap();
  await page.getByRole('tab', { name: 'Benchmarks', exact: true }).tap();
  await ensureGridBenchmark(page, 'Fictional GEMM');
  const dialog = await inspectResult(page, 'Fictional GEMM', 'fictional-develop-23');
  await expect(dialog.getByText('Run Provenance', { exact: true })).toBeVisible();
  await expectDialogTypographyContained(dialog);
  const box = await dialog.boundingBox();
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(page.viewportSize().width);
  await dialog.getByRole('button', { name: 'Close details', exact: true }).tap();
  await expect(dialog).toHaveCount(0);
  const inspector = page.locator('[role="dialog"][aria-labelledby="inspect-results-title"]');
  await expect(inspector).toBeVisible();
  await inspector.getByRole('button', { name: 'Close', exact: true }).tap();
  await expect(inspector).toHaveCount(0);
  await expectNoDocumentOverflow(page);
  await page.getByRole('tab', { name: 'Run Comparison', exact: true }).tap();
  await expectLocalScroll(page.getByRole('region', { name: 'Compared run metadata scroll area' }), { horizontal: true });
  await expectLocalScroll(page.getByRole('region', { name: 'Compared run metadata scroll area' }));
  await expectNoDocumentOverflow(page);
  expect(errors).toEqual([]);
});

test('phone branch list/detail restores local scroll, focus and exact selections with Back and Forward', async ({ page }) => {
  await openDashboard(page, '/?view=branch');
  await expect(page.getByRole('button', { name: /^Filters ·/ })).toHaveCount(0);
  const view = page.getByTestId('branch-runs');
  const list = page.getByTestId('branch-list-scroll');
  await expect(view).toHaveAttribute('data-screen', 'list');
  await expect(page.locator('#branch-detail')).toBeHidden();
  await expectLocalScroll(list);
  const listPosition = await list.evaluate((element) => element.scrollTop);
  const selected = page.getByRole('button', { name: 'View branch fictional/optimization-01, PR #1000', exact: true });
  await selected.tap();
  await expect(view).toHaveAttribute('data-screen', 'detail');
  await expect(list).toBeHidden();
  await expect(page.getByRole('heading', { name: 'fictional/optimization-01', exact: true })).toBeFocused();
  await expect(page.getByTestId('branch-mobile-results')).toBeVisible();
  await expect(page.getByRole('table', { name: 'Branch benchmark differences', exact: true })).toBeHidden();
  await page.getByTestId('branch-config-gfx950-MT').tap();
  await expect(page.getByTestId('branch-selected-configuration')).toHaveText('gfx950MT');
  await expectNoDocumentOverflow(page);
  await page.getByRole('button', { name: '← Back to branches', exact: true }).tap();
  await expect(view).toHaveAttribute('data-screen', 'list');
  await expect(selected).toBeFocused();
  await expect.poll(() => list.evaluate((element) => element.scrollTop)).toBeCloseTo(listPosition, 0);
  await page.goBack();
  await expect(view).toHaveAttribute('data-screen', 'detail');
  await expect(page.getByTestId('candidate-selected-identity')).toContainText('fictional-branch-01');
  await expect(page.getByTestId('branch-config-gfx950-MT')).toHaveAttribute('aria-pressed', 'true');
  await page.goForward();
  await expect(view).toHaveAttribute('data-screen', 'list');
  await expect.poll(() => list.evaluate((element) => element.scrollTop)).toBeCloseTo(listPosition, 0);
  await expectNoDocumentOverflow(page);
});
