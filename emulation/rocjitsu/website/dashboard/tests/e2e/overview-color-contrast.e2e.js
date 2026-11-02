import { expect, test } from '@playwright/test';
import { readChart } from './helpers/chart.js';
import { openDashboard } from './helpers/dashboard.js';

for (const width of [1440, 390]) test(`Overview configuration colors are readable and match chart identities at ${width}px`, async ({ page }) => {
  await page.setViewportSize({ width, height: 1000 });
  await openDashboard(page);
  for (const theme of ['light', 'dark']) {
    if (theme === 'dark') await page.getByRole('button', { name: 'Use dark theme', exact: true }).click();
    const values = await page.getByTestId('metric-card-latest-duration').evaluate((card) => {
      const luminance = (color) => color.match(/[\d.]+/g).slice(0, 3).map(Number).map((value) => value / 255)
        .map((value) => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4)
        .reduce((total, value, index) => total + value * [0.2126, 0.7152, 0.0722][index], 0);
      const background = luminance(getComputedStyle(card).backgroundColor);
      return [...card.querySelectorAll('[data-configuration]')].map((row) => {
        const color = getComputedStyle(row.lastElementChild).color;
        const foreground = luminance(color);
        return { name: row.dataset.configuration.replace('/', ' '), color,
          contrast: (Math.max(foreground, background) + 0.05) / (Math.min(foreground, background) + 0.05) };
      });
    });
    expect(values).toHaveLength(4);
    for (const value of values) expect(value.contrast, `${theme} ${value.name}`).toBeGreaterThanOrEqual(4.5);
    const trend = page.getByTestId('performance-trend');
    const lines = await readChart(trend.getByRole('img'), (chart) => chart.getOption().series.map((series) => ({ name: series.name, color: series.lineStyle.color })));
    for (const value of values) {
      const line = lines.find((line) => line.name === value.name);
      expect(line).toBeDefined();
      const normalized = await page.evaluate((hex) => { const node = document.createElement('span'); node.style.color = hex; document.body.append(node); const color = getComputedStyle(node).color; node.remove(); return color; }, line.color);
      expect(normalized).toBe(value.color);
      await expect(trend.getByText(new RegExp(`^[●◆]${value.name}$`)).locator('span')).toHaveCSS('color', value.color);
    }
    const inspection = trend.getByRole('group', { name: 'Inspect performance trend', exact: true });
    await inspection.press('End');
    const overlay = trend.getByTestId('hover-line-focus');
    await expect(overlay).toBeVisible();
    await expect(overlay.locator('path')).toHaveCount(lines.length);
    expect(await overlay.locator('path').evaluateAll((paths) => paths.map((path) => path.getAttribute('stroke')).sort()))
      .toEqual(lines.map((line) => line.color).sort());
    await inspection.press('Escape');
  }
});
