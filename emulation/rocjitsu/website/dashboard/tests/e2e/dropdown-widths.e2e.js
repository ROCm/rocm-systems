import { expect, test } from '@playwright/test';
import { openDashboard } from './helpers/dashboard.js';

async function expectAlignedPopup(page, picker, resizeWidths = []) {
  const field = await picker.elementHandle();
  await picker.click();
  const list = page.getByRole('listbox');
  await expect(list).toBeVisible();
  const primary = list.getByRole('option').first();
  for (const width of [null, ...resizeWidths]) {
    if (width) await page.setViewportSize({ width, height: 900 });
    await expect.poll(() => primary.evaluate((node) => {
      const text = node.querySelector('.MuiTypography-body2, .MuiListItemText-primary') ?? node;
      const scale = Number(getComputedStyle(document.body).zoom);
      return parseFloat(getComputedStyle(text).fontSize) - 14 * scale;
    })).toBeCloseTo(0, 1);
    await expect.poll(async () => {
    const anchor = await field.evaluate((node) => {
      const rect = node.closest('.MuiInputBase-root').getBoundingClientRect();
      return { left: rect.left, width: rect.width };
    });
    const popup = await list.evaluate((node) => {
      const rect = node.closest('.MuiPaper-root').getBoundingClientRect();
      return { left: rect.left, width: rect.width, right: rect.right };
    });
    return Math.max(Math.abs(anchor.width - popup.width), Math.abs(anchor.left - popup.left));
    }).toBeLessThan(2);
  }
  await page.keyboard.press('Escape');
  await expect(list).toBeHidden();
}

for (const width of [1440, 900, 390]) {
  test(`branch dropdowns match their fields at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await openDashboard(page, '/?view=branch&detail=1');
    for (const label of ['Candidate', 'Reference', 'Suites']) {
      await expectAlignedPopup(page, page.getByRole('combobox', { name: label, exact: true }), width === 1440 ? [899, 1440] : []);
    }
  });

  test(`comparison and catalog dropdowns match their fields at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await openDashboard(page, '/?view=compare');
    for (const label of ['Baseline run', 'Candidate run']) {
      await expectAlignedPopup(page, page.getByRole('combobox', { name: label, exact: true }), width === 1440 ? [899, 1440] : []);
    }
    await openDashboard(page, '/?view=overview');
    await expectAlignedPopup(page, page.getByRole('combobox', { name: 'Benchmark set', exact: true }));
  });

  test(`benchmark dropdowns match their fields at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await openDashboard(page, '/?view=benchmarks');
    await page.getByRole('button', { name: 'Add benchmarks', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'Add benchmarks', exact: true });
    await expect(dialog).toBeVisible();
    await expectAlignedPopup(page, page.getByRole('combobox', { name: 'Benchmarks to graph' }));
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(dialog).toBeHidden();
    await page.getByRole('button', { name: 'Inspect Fictional GEMM results', exact: true }).click();
    await expectAlignedPopup(page, page.getByRole('dialog').getByRole('combobox'));
  });
}
