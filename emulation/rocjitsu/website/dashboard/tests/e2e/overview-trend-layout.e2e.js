import { expect, test } from '@playwright/test';
import { readChart } from './helpers/chart.js';

// Use the checked-in fictional publication served by the fixtures preview.
for (const range of ['1W', '1M', '3M', 'ALL']) {
  test(`${range} respects its time window and shows a non-solid reference line for fictional slowdowns`, async ({ page }) => {
    await page.goto(`/?view=overview&range=${range}`);
    const trend = page.getByTestId('performance-trend');
    const chart = trend.getByRole('img');
    const geometry = await readChart(chart, (instance) => {
      const model = instance.getModel();
      const rect = model.getComponent('grid').coordinateSystem.getRect();
      const option = instance.getOption();
      const first = option.series[0].data.find((point) => Number.isFinite(point[1]));
      const pixel = instance.convertToPixel({ seriesIndex: 0 }, first);
      return {
        left: pixel[0] - rect.x,
        baseline: instance.convertToPixel({ yAxisIndex: 0 }, 1) - pixel[1],
        belowBaseline: option.series.some((series) => series.data.some((point) => Number.isFinite(point[1]) && point[1] < 1)),
        reference: option.series[0].markLine,
        axis: option.xAxis[0].axisLine,
        min: option.xAxis[0].min, max: option.xAxis[0].max,
        firstDateX: instance.convertToPixel({ xAxisIndex: 0 }, model.getComponent('xAxis').axis.getViewLabels()[0].tick.value),
        datePixels: model.getComponent('xAxis').axis.getViewLabels().map((label) => instance.convertToPixel({ xAxisIndex: 0 }, label.tick.value)),
        pixel, width: instance.getWidth(), height: instance.getHeight(),
      };
    });
    if (range === '1M' || range === '3M') {
      expect(geometry.min).toBe(0);
      expect(geometry.max).toBe((range === '1M' ? 30 : 90) - 1);
      expect(geometry.left).toBeGreaterThan(0);
    } else {
      expect(Math.abs(geometry.left)).toBeLessThan(1);
    }
    expect(Math.abs(geometry.baseline)).toBeLessThan(1);
    expect(geometry.belowBaseline).toBe(true);
    expect(geometry.reference.lineStyle.type).not.toBe('solid');
    expect(geometry.reference.data).toMatchObject([{ yAxis: 1 }]);
    expect(geometry.axis).toMatchObject({ show: true, onZero: false, lineStyle: { type: 'solid' } });
    if (range === '1W') expect(Math.abs(geometry.firstDateX - geometry.pixel[0])).toBeLessThan(1);
    const step = geometry.datePixels[1] - geometry.datePixels[0];
    expect(step).toBeGreaterThan(0);
    for (let index = 2; index < geometry.datePixels.length; index += 1) {
      expect(Math.abs(geometry.datePixels[index] - geometry.datePixels[index - 1] - step)).toBeLessThan(1);
    }
    const box = await chart.boundingBox();
    await page.mouse.move(box.x + geometry.pixel[0] * box.width / geometry.width, box.y + geometry.pixel[1] * box.height / geometry.height);
    await expect(chart.getByText(/^Commit [a-f0-9]{8}$/)).toBeVisible();
    await expect(chart.getByText(/^Run time:/)).toHaveCount(0);
    const tooltip = await chart.locator('.dashboard-chart-tooltip').innerText();
    expect(tooltip).not.toContain('Attempt:');
    expect(tooltip).not.toContain('Reference:');
    expect(tooltip).toMatch(/gfx1250 single · [\d.]+× · (?:\d+m )?\d+\.\d{1,2}s · ref (?:\d+m )?\d+\.\d{1,2}s/);
  });
}

// The tooltip is as wide as its longest row, so on a narrow chart it must be capped at the chart width
// or the card clips the end of the reference time.
for (const width of [390, 900]) {
  test(`the trend tooltip stays inside the chart at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/?view=overview&range=ALL');
    const chart = page.getByTestId('performance-trend').getByRole('img');
    const last = await readChart(chart, (instance) => {
      const option = instance.getOption();
      let best = null;
      option.series.forEach((series, seriesIndex) => {
        if (series.type !== 'line') return;
        for (const point of series.data ?? []) {
          const value = point && (point.value ?? point);
          if (!Array.isArray(value) || !Number.isFinite(value[1])) continue;
          const pixel = instance.convertToPixel({ seriesIndex }, value);
          if (!best || pixel[0] > best.pixel[0]) best = { pixel };
        }
      });
      return { pixel: best.pixel, width: instance.getWidth(), height: instance.getHeight() };
    });
    await chart.scrollIntoViewIfNeeded();
    const box = await chart.boundingBox();
    await page.mouse.move(box.x + last.pixel[0] * box.width / last.width, box.y + last.pixel[1] * box.height / last.height);
    const tooltip = chart.locator('.performance-trend-tooltip');
    await expect(tooltip).toBeVisible();
    await expect(tooltip).toContainText(/ref (?:\d+m )?\d+\.\d{1,2}s/);
    const overflow = await tooltip.evaluate((element) => {
      const tip = element.getBoundingClientRect();
      const frame = element.parentElement.getBoundingClientRect();
      return { right: tip.right - frame.right, left: frame.left - tip.left };
    });
    expect(overflow.right).toBeLessThanOrEqual(0.5);
    expect(overflow.left).toBeLessThanOrEqual(0.5);
  });
}

test('an all-above-baseline configuration has a solid bottom axis and keeps the baseline reference line', async ({ page }) => {
  await page.goto('/?view=overview&range=1W');
  const trend = page.getByTestId('performance-trend');
  await expect(trend.getByRole('img')).toBeVisible();
  await page.getByRole('checkbox', { name: 'gfx950', exact: true }).uncheck();
  await page.getByRole('checkbox', { name: 'single', exact: true }).uncheck();
  const chart = trend.getByRole('img');
  await expect.poll(() => readChart(chart, (instance) => instance.getOption().series.length)).toBe(1);
  const option = await readChart(chart, (instance) => {
    const option = instance.getOption();
    return { reference: option.series[0].markLine?.data ?? [], axis: option.xAxis[0].axisLine };
  });
  expect(option.reference).toMatchObject([{ yAxis: 1 }]);
  expect(option.axis).toMatchObject({ show: true, onZero: false, lineStyle: { type: 'solid' } });
});

test('closed catalog disclosure expands and collapses', async ({ page }) => {
  await page.goto('/?view=overview&range=1W');
  const disclosure = page.getByTestId('catalog-workloads');
  await expect(disclosure).toBeVisible();
  await disclosure.locator('summary').click();
  await expect(disclosure.getByRole('region', { name: 'Catalog workload membership' })).toBeVisible();
  await disclosure.locator('summary').click();
  await expect(disclosure.getByRole('region', { name: 'Catalog workload membership' })).toBeHidden();
});
