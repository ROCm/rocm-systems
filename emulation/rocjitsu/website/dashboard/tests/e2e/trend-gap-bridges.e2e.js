import { expect, test } from '@playwright/test';
import { readChart } from './helpers/chart.js';
import { openDashboard } from './helpers/dashboard.js';

test('trend bridges are dashed endpoint-only decoration in both themes', async ({ page }) => {
  await openDashboard(page);
  const trend = page.getByTestId('performance-trend');
  const chart = trend.getByRole('img');
  for (const theme of ['light', 'dark']) {
    if (theme === 'dark') await page.getByRole('button', { name: 'Use dark theme' }).click();
    const result = await readChart(chart, (instance) => {
      const option = instance.getOption();
      const [solid, ...bridges] = option.series;
      const expected = [];
      let previous = null;
      solid.data.forEach((point, index) => {
        if (!Number.isFinite(point[1])) return;
        if (previous != null && index > previous + 1) expected.push([solid.data[previous], point]);
        previous = index;
      });
      return { connectNulls: solid.connectNulls, expected, bridges: bridges.map((bridge) => ({
        data: bridge.data, connectNulls: bridge.connectNulls, silent: bridge.silent, symbol: bridge.symbol, showSymbol: bridge.showSymbol,
        type: bridge.lineStyle.type, color: bridge.lineStyle.color, area: bridge.areaStyle,
        tooltip: bridge.tooltip.show,
      })), color: solid.lineStyle.color,
      fauxTooltip: option.tooltip[0].formatter([{ seriesIndex: 1, dataIndex: 0, value: bridges[0]?.data[0] }]) };
    });
    expect(result.connectNulls).toBe(false);
    expect(result.expected.length).toBeGreaterThan(0);
    expect(result.bridges).toHaveLength(1);
    expect(result.bridges[0].connectNulls).toBe(false);
    const segments = [[]];
    for (const point of result.bridges[0].data) {
      if (Number.isFinite(point[1])) segments.at(-1).push(point);
      else segments.push([]);
    }
    expect(segments).toEqual(result.expected);
    for (const bridge of result.bridges) {
      expect(bridge).toMatchObject({ silent: true, symbol: 'none', showSymbol: false, type: 'dashed', color: result.color, tooltip: false });
      expect(bridge.area).toBeUndefined();
    }
    expect(result.fauxTooltip).toBe('');
    const group = trend.getByRole('group', { name: 'Inspect performance trend' });
    await group.press('Home');
    await expect(trend.getByRole('status')).not.toHaveText(/—/);
    await group.press('ArrowRight');
    await expect(trend.getByRole('status')).not.toHaveText(/—/);
    await group.press('Escape');
  }
});
