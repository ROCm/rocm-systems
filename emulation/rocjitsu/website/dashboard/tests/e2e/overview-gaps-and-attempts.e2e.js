import { expect, test } from '@playwright/test';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { readChart } from './helpers/chart.js';
import { installPublication, openDashboard } from './helpers/dashboard.js';

test('daily gaps hide tooltip and latest failures never fall back to earlier success', async ({ page }) => {
  const publication = createMaintainedPublication();
  const latest = publication.runs.find((r) => r.publication.id === 'fictional-develop-23');
  for (const config of latest.benchmark_results.filter(({ id }) => id === 'a')) {
    config.status = 'failed'; config.error = 'Fictional latest failure';
  }
  await installPublication(page, { publication });
  await openDashboard(page, '/?range=1M');
  await expect(page.getByTestId('metric-card-latest-duration')).toContainText('Unavailable');
  await expect(page.getByTestId('metric-card-relative-speedup')).toContainText('Unavailable');
  await expect(page.getByTestId('largest-changes')).toContainText('failed');
  const trend = page.getByTestId('performance-trend');
  const chart = trend.getByRole('img');
  const probe = await readChart(chart, (instance) => {
    const series = instance.getOption().series;
    const gap = series[0].data.findIndex((_, i) => series.every((s) => s.data[i][1] === null));
    return { gap, x: instance.convertToPixel({ xAxisIndex: 0 }, series[0].data[gap][0]),
      y: 80, width: instance.getWidth(), height: instance.getHeight(),
      tooltip: instance.getOption().tooltip[0].formatter([{ dataIndex: gap }]),
      connect: series.map((s) => s.connectNulls) };
  });
  expect(probe.gap).toBeGreaterThanOrEqual(0); expect(probe.tooltip).toBe(''); expect(probe.connect.every((v) => v === false)).toBe(true);
  const box = await chart.boundingBox();
  await page.mouse.move(box.x + probe.x * box.width / probe.width, box.y + probe.y * box.height / probe.height);
  await expect(trend.getByTestId('point-hover-guide')).not.toBeVisible();
});

test('weekly repeated attempts preserve collision-safe inspection and touch disclosure', async ({ page }) => {
  const publication = createMaintainedPublication();
  const attempt = structuredClone(publication.runs.find((r) => r.publication.id === 'fictional-develop-23'));
  attempt.publication.id = 'fictional-repeat'; attempt.execution_summary.finished_at = '2026-10-04T07:00:00.000Z';
  publication.runs.push(attempt); publication.index.runFiles.push('runs/default-branch/fictional-repeat.json');
  await installPublication(page, { publication }); await openDashboard(page, '/?range=1W');
  const trend = page.getByTestId('performance-trend'); const group = trend.getByRole('group', { name: 'Inspect performance trend' });
  await group.press('End'); await expect(trend.getByRole('status')).toContainText('fictional-repeat');
  await group.press('ArrowLeft'); await expect(trend.getByRole('status')).toContainText('fictional-develop-23');
  const chart = trend.getByRole('img');
  const probe = await readChart(chart, (instance) => ({ pixel: instance.convertToPixel({ seriesIndex: 0 }, instance.getOption().series[0].data.at(-1)), width: instance.getWidth(), height: instance.getHeight() }));
  const box = await chart.boundingBox();
  await group.dispatchEvent('pointerdown', { pointerType: 'touch', clientX: box.x + probe.pixel[0] * box.width / probe.width, clientY: box.y + probe.pixel[1] * box.height / probe.height });
  await group.dispatchEvent('pointerleave', { pointerType: 'touch' });
  await expect(trend.getByRole('status')).toContainText('fictional-repeat');
  await group.press('Escape'); await expect(trend.getByRole('status')).toHaveCount(0);
});
