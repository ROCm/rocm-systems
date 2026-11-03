import { expect, test } from '@playwright/test';
import { readChart } from './helpers/chart.js';

// Use the checked-in fictional publication served by the fixtures preview.
for (const range of ['1W', '1M', '3M', 'ALL']) {
  test(`${range} respects its time window and shows a dotted reference for fictional slowdowns`, async ({ page }) => {
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
    expect(geometry.reference.lineStyle.type).toBe('dotted');
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
    const tooltip = await chart.getByText(/^Commit [a-f0-9]{8}$/).evaluate((heading) => heading.parentElement.innerText);
    expect(tooltip).not.toContain('Attempt:');
    expect(tooltip).not.toContain('Reference:');
    expect(tooltip).toMatch(/gfx1250 single · [\d.]+× · [\d.]+s · ref [\d.]+s/);
  });
}

test('an all-above-baseline configuration has a solid bottom axis without a reference overlay', async ({ page }) => {
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
  expect(option.reference).toEqual([]);
  expect(option.axis).toMatchObject({ show: true, onZero: false, lineStyle: { type: 'solid' } });
});

test('closed catalog disclosure keeps the latest-run note close without breaking expansion', async ({ page }) => {
  await page.goto('/?view=overview&range=1W');
  const disclosure = page.getByTestId('catalog-workloads');
  await expect(disclosure).toBeVisible();
  const note = page.getByTestId('performance-trend').getByText(/^Latest eligible plotted run/);
  await expect(note).toBeVisible();
  const gap = await note.evaluate((element) => {
    const summary = document.querySelector('[data-testid="catalog-workloads"] summary');
    const scale = summary.getBoundingClientRect().width / summary.offsetWidth;
    return (element.getBoundingClientRect().top - summary.getBoundingClientRect().bottom) / scale;
  });
  expect(gap).toBeLessThanOrEqual(5);
  await disclosure.locator('summary').click();
  await expect(disclosure.getByRole('region', { name: 'Catalog workload membership' })).toBeVisible();
  await disclosure.locator('summary').click();
  await expect(disclosure.getByRole('region', { name: 'Catalog workload membership' })).toBeHidden();
});
