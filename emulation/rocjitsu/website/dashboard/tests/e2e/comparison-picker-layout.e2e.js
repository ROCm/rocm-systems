import { expect, test } from '@playwright/test';
import { openDashboard, selectRun } from './helpers/dashboard.js';

for (const width of [1439, 390]) {
  test(`comparison controls stay fixed from empty to selected and swapped at ${width}px`, async ({ browser }) => {
    const context = await browser.newContext({
      viewport: { width, height: 643 }, screen: { width: 1835, height: 768 }, deviceScaleFactor: 1.875,
    });
    try {
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await openDashboard(page, '/?view=compare');
      const controls = [
        page.getByRole('combobox', { name: 'Candidate run', exact: true }),
        page.getByRole('button', { name: 'Swap', exact: true }),
        page.getByRole('combobox', { name: 'Baseline run', exact: true }),
      ];
      const geometry = () => Promise.all(controls.map((control) => control.evaluate((element) => {
        const rect = element.getBoundingClientRect();
        return { x: rect.left + scrollX, y: rect.top + scrollY };
      })));
      const original = await geometry();
      const expectStable = async () => {
        await expect.poll(async () => Math.max(...(await geometry()).map((position, index) =>
          Math.max(Math.abs(position.x - original[index].x), Math.abs(position.y - original[index].y)))))
          .toBeLessThan(1);
      };
      await expect(controls[1]).toBeDisabled();
      await selectRun(page, 'Candidate run', 'fictional-develop-23');
      await expect(page.getByTestId('candidate-run-selected-identity')).toContainText('fictional-develop-23');
      await expectStable();
      await selectRun(page, 'Baseline run', 'fictional-develop-17');
      await expect(page.getByTestId('baseline-run-selected-identity')).toContainText('fictional-develop-17');
      await expectStable();
      await controls[1].click();
      await expect(page.getByTestId('candidate-run-selected-identity')).toContainText('fictional-develop-17');
      await expect(page.getByTestId('baseline-run-selected-identity')).toContainText('fictional-develop-23');
      await expectStable();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      expect(errors).toEqual([]);
    } finally {
      await context.close();
    }
  });
}
