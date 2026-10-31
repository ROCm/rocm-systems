import { expect, test } from '@playwright/test';
import { expectNoDocumentOverflow, openDashboard } from './helpers/dashboard.js';

for (const width of [1440, 390]) {
  test(`branch footer and configuration labels fit at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await openDashboard(page, '/?view=branch&detail=1');
    const matrix = page.getByRole('region', { name: 'Change by configuration' });
    for (const label of ['single', 'default']) {
      const tag = matrix.locator('[data-category="mode"]').filter({ hasText: label });
      await expect(tag).toBeVisible();
      expect(await tag.evaluate((node) => {
        const text = node.querySelector('.MuiChip-label');
        return text.scrollWidth <= text.clientWidth + 1 && node.getBoundingClientRect().width <= node.parentElement.getBoundingClientRect().width + 1;
      })).toBe(true);
    }
    if (width === 1440) {
      const note = await page.getByText(/^Negative change means less measured wall time/).boundingBox();
      const button = await page.getByTestId('branch-open-comparison').boundingBox();
      expect(button.y).toBeLessThan(note.y + note.height);
      expect(note.y).toBeLessThan(button.y + button.height);
    }
    await expectNoDocumentOverflow(page);
    const results = width === 1440 ? page.getByRole('table', { name: 'Benchmark differences', exact: true }) : page.getByTestId('branch-mobile-results');
    const suiteLabels = results.locator('[data-category="suite"]');
    await expect(suiteLabels).toHaveText(['Llama', 'Triton']);
    expect(await results.getByText(/^[a-e]$/, { exact: true }).count()).toBeGreaterThan(0);
    const sortId = width === 1440 ? 'branch-sort-percent' : 'branch-mobile-sort-percent';
    await page.getByTestId(sortId).click();
    await expect(suiteLabels).toHaveText(['Llama', 'Triton']);
  });
}

test('comparison displays candidate before baseline without swapping the selected identities', async ({ page }) => {
  await openDashboard(page, '/?view=compare&compareCandidate=fictional-branch-01&compareBaseline=fictional-develop-20');
  const candidate = page.getByRole('combobox', { name: 'Candidate run', exact: true });
  const baseline = page.getByRole('combobox', { name: 'Baseline run', exact: true });
  expect((await candidate.boundingBox()).x).toBeLessThan((await baseline.boundingBox()).x);
  await expect(page.getByTestId('candidate-run-selected-identity')).toContainText('fictional-branch-01');
  await expect(page.getByTestId('baseline-run-selected-identity')).toContainText('fictional-develop-20');
  await page.getByRole('button', { name: 'Swap', exact: true }).click();
  await expect(page.getByTestId('candidate-run-selected-identity')).toContainText('fictional-develop-20');
  await expect(page.getByTestId('baseline-run-selected-identity')).toContainText('fictional-branch-01');
});
