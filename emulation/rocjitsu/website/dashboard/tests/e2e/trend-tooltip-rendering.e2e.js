import { expect, test } from '@playwright/test';
import { readChart } from './helpers/chart.js';

for (const theme of ['light', 'dark']) {
  test(`trend tooltips avoid scaled compositor layers at different points in ${theme} theme`, async ({ browser }) => {
    const context = await browser.newContext({
      viewport: { width: 1439, height: 643 }, screen: { width: 1835, height: 768 }, deviceScaleFactor: 1.875,
    });
    try {
      const page = await context.newPage();
      await page.goto('/?view=overview&range=ALL');
      if (theme === 'dark') await page.getByRole('button', { name: 'Use dark theme', exact: true }).click();
      const trend = page.getByTestId('performance-trend');
      const chart = trend.getByRole('img');
      await readChart(chart, (instance) => instance.getOption().series.length);
      const group = trend.getByRole('group', { name: 'Inspect performance trend', exact: true });
      const positions = [];
      for (const steps of [0, 10, 19]) {
        await group.press('End');
        for (let index = 0; index < steps; index += 1) await group.press('ArrowLeft');
        const heading = chart.getByText(/^Commit [a-f0-9]{8}$/);
        await expect(heading).toBeVisible();
        positions.push(await heading.evaluate((element) => {
          const tooltip = element.parentElement;
          const style = getComputedStyle(tooltip);
          const rect = tooltip.getBoundingClientRect();
          const chart = tooltip.parentElement.getBoundingClientRect();
          return { transform: style.transform, willChange: style.willChange, backdropFilter: style.backdropFilter,
            x: rect.x * devicePixelRatio, y: rect.y * devicePixelRatio,
            inside: rect.left >= chart.left - 1 && rect.right <= chart.right + 1 && rect.top >= chart.top - 1 && rect.bottom <= chart.bottom + 1 };
        }));
      }

      for (const position of positions) {
        expect(position.transform).toBe('none');
        expect(position.willChange).toBe('auto');
        expect(position.backdropFilter).toBe('none');
        expect(Math.abs(position.x - Math.round(position.x))).toBeLessThan(0.04);
        expect(Math.abs(position.y - Math.round(position.y))).toBeLessThan(0.04);
        expect(position.inside).toBe(true);
      }
      await group.press('Escape');
      await expect(chart.getByText(/^Commit [a-f0-9]{8}$/)).toBeHidden();
    } finally {
      await context.close();
    }
  });
}
