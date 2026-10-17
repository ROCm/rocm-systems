import { expect, test } from '@playwright/test';
import { createRecentRunsHistory } from '../fixtures/recent-runs-history.js';
import { installPublication, openDashboard, ready, expectNoDocumentOverflow } from './helpers/dashboard.js';

for (const width of [1440, 390]) {
  test(`Recent Runs pages through every validated attempt at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const { publication, data } = createRecentRunsHistory();
    expect(data.runs).toHaveLength(65);
    await installPublication(page, { publication });
    await openDashboard(page);
    const card = page.getByTestId('recent-runs');
    const rows = card.locator('tbody tr');
    const older = card.getByRole('button', { name: 'Go to next page', exact: true });
    const newer = card.getByRole('button', { name: 'Go to previous page', exact: true });
    const expected = publication.runs.filter((run) => run.source.branch === 'develop')
      .sort((a, b) => b.execution.completedAt.localeCompare(a.execution.completedAt) || b.id.localeCompare(a.id)).map((run) => run.id);
    const first = card.getByRole('button', { name: 'Go to first page', exact: true });
    const last = card.getByRole('button', { name: 'Go to last page', exact: true });
    await expect(newer).toBeDisabled();
    await expect(first).toBeDisabled();
    for (const control of [first, newer, older, last]) {
      await expect(control).toHaveText('');
      await expect(control.locator('svg')).toHaveCount(1);
      await expect(control).toHaveAttribute('title', await control.getAttribute('aria-label'));
    }
    await last.click();
    await expect(card.getByRole('status')).toHaveText('Showing 61–65 of 65 runs');
    await expect(last).toBeDisabled();
    await expect(older).toBeDisabled();
    await first.click();
    await expect(card.getByRole('status')).toHaveText('Showing 1–20 of 65 runs');
    await card.getByRole('button', { name: 'Go to page 3', exact: true }).click();
    await expect(card.getByRole('status')).toHaveText('Showing 41–60 of 65 runs');
    await expect(card.getByRole('button', { name: 'page 3', exact: true })).toHaveAttribute('aria-current', 'page');
    await card.getByRole('button', { name: 'Go to page 1', exact: true }).click();
    const seen = [];
    for (let index = 0; index < 4; index++) {
      const slice = expected.slice(index * 20, index * 20 + 20);
      await expect(card.getByRole('status')).toHaveText(`Showing ${index * 20 + 1}–${index * 20 + slice.length} of 65 runs`);
      await expect(rows).toHaveCount(slice.length);
      const ids = await rows.evaluateAll((elements) => elements.map((element) => element.dataset.runId));
      expect(ids).toEqual(slice);
      seen.push(...ids);
      if (index < 3) {
        await card.getByTestId('recent-runs-table').evaluate((element) => { element.scrollTop = element.scrollHeight; });
        await older.focus();
        await older.press('Enter');
        await expect.poll(() => card.getByTestId('recent-runs-table').evaluate((element) => element.scrollTop)).toBe(0);
      }
    }
    expect(seen).toEqual(expected);
    await expect(older).toBeDisabled();
    await newer.click();
    await expect(card.getByRole('status')).toHaveText('Showing 41–60 of 65 runs');
    await expect(rows.first()).toHaveAttribute('data-run-id', expected[40]);
    await expectNoDocumentOverflow(page);
  });
}

test('Recent Runs refresh recovers from shrinking and empty histories', async ({ page }) => {
  await installPublication(page, { publication: createRecentRunsHistory().publication });
  await openDashboard(page);
  const card = page.getByTestId('recent-runs');
  await card.getByRole('button', { name: 'Go to page 4', exact: true }).click();
  await expect(card.getByRole('status')).toHaveText('Showing 61–65 of 65 runs');
  for (const count of [24, 3, 0, 65]) {
    await installPublication(page, { publication: createRecentRunsHistory(count).publication });
    await page.getByRole('button', { name: 'Reload all data', exact: true }).click();
    await ready(page);
    if (count === 0) {
      await expect(card).toContainText('No runs available.');
      await expect(card.locator('tbody tr')).toHaveCount(0);
      await expect(card.getByRole('button', { name: 'Go to next page', exact: true })).toHaveCount(0);
    } else {
      // A full reload may remount the view; either first or last valid page is safe.
      await expect(card.locator('tbody tr').first()).toBeVisible();
      await expect(card.getByRole('status')).toHaveText(count === 24 ? /Showing (1–20|21–24) of 24 runs/ : `Showing 1–${Math.min(count, 20)} of ${count} runs`);
      if (count === 3) {
        await expect(card.getByRole('button', { name: 'Go to next page', exact: true })).toBeDisabled();
        await expect(card.getByRole('button', { name: 'Go to previous page', exact: true })).toBeDisabled();
        await expect(card.getByRole('button', { name: 'Go to first page', exact: true })).toBeDisabled();
        await expect(card.getByRole('button', { name: 'Go to last page', exact: true })).toBeDisabled();
      }
    }
  }
});

test('long Recent Runs pagination collapses pages and supports a direct last-page jump', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  await installPublication(page, { publication: createRecentRunsHistory(205).publication });
  await openDashboard(page);
  const card = page.getByTestId('recent-runs');
  const pages = card.getByRole('navigation', { name: 'Recent runs pages' });
  await expect(pages).toContainText('…');
  expect(await pages.getByRole('button').count()).toBeLessThan(11);
  await pages.getByRole('button', { name: 'Go to last page', exact: true }).click();
  await expect(card.getByRole('status')).toHaveText('Showing 201–205 of 205 runs');
  await expect(card.locator('tbody tr')).toHaveCount(5);
  await expectNoDocumentOverflow(page);
});
