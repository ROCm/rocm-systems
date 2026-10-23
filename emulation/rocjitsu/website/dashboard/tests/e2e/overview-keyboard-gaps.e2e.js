import { expect, test } from '@playwright/test';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { downloadSource, installPublication, ready } from './helpers/dashboard.js';
import { readChart } from './helpers/chart.js';

function keyboardPublication() {
  const original = createMaintainedPublication();
  const latest = original.runs.find((run) => run.publication.id === 'fictional-develop-23');
  const dates = ['2026-09-30', '2026-10-01', '2026-10-02', '2026-10-03'];
  const runs = dates.map((day, index) => {
    const run = structuredClone(latest);
    run.publication.id = `qa-keyboard-${index}`;
    run.provenance.rocjitsu.rocjitsu_commit_sha = String(index + 1).repeat(40);
    run.provenance.rocjitsu.rocjitsu_commit_timestamp = `${day}T10:00:00Z`;
    run.execution_summary.timestamp = `${day}T11:00:00Z`;
    run.execution_summary.finished_at = `${day}T12:00:00Z`;
    if (index === 1 || index === 2) {
      for (const configuration of run.benchmark_results) {
        if (index === 2 && configuration.target !== 'gfx1250') continue;
        Object.assign(configuration, { status: 'timeout', timing_results_s: [], error: 'QA-only keyboard timeout' });
      }
    }
    return run;
  });
  return { index: { ...original.index, runFiles: runs.map((run) => `runs/default-branch/${run.publication.id}.json`) },
    runs };
}

for (const width of [1440, 390]) test(`keyboard slots expose truthful gaps and preserve comparisons at ${width}px`, async ({ page }, testInfo) => {
  await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  const publication = keyboardPublication();
  const last = publication.runs.at(-1);
  const savedLast = structuredClone(last);
  await installPublication(page, { publication });
  await page.goto('/?targets=gfx1250&targets=gfx950&modes=single&range=1M');
  await ready(page);
  const trend = page.getByTestId('performance-trend');
  const group = trend.getByRole('group', { name: 'Inspect performance trend', exact: true });
  const status = trend.getByTestId('trend-inspection-status');
  const cards = page.getByTestId('metric-strip');
  const changes = page.getByTestId('largest-changes');
  const originalCards = await cards.innerText();
  const originalChanges = await changes.innerText();
  const chart = trend.getByRole('img');
  await readChart(chart, (instance) => instance.getOption().series.length);
  await group.scrollIntoViewIfNeeded();
  await trend.getByRole('combobox', { name: 'Benchmark set', exact: true }).focus();
  await page.keyboard.press('Tab');
  await expect(group).toBeFocused();
  await page.keyboard.press('End');
  await expect(status).toBeVisible();
  await expect(status).toContainText('qa-keyboard-3');
  await expect(status).toHaveAttribute('data-visible', 'true');
  const tooltipVisible = () => readChart(chart, (instance) => [...instance.getDom().querySelectorAll('.performance-trend-tooltip')]
    .some((element) => getComputedStyle(element).display !== 'none' && getComputedStyle(element).visibility !== 'hidden'
      && getComputedStyle(element).opacity !== '0'));
  await expect.poll(tooltipVisible).toBe(true);
  await page.keyboard.press('ArrowLeft');
  await expect(status).toContainText('qa-keyboard-2');
  await expect(status).toContainText('gfx1250 single: Unavailable — Timed out');
  await expect(status).toContainText('gfx950 single: 1.000×');
  await page.keyboard.press('ArrowLeft');
  await expect(status).toContainText('2026-10-01 UTC');
  await expect(status).toContainText('qa-keyboard-1');
  await expect(trend.getByTestId('point-hover-guide')).not.toBeVisible();
  await expect(trend.getByTestId('hover-line-focus')).not.toBeVisible();
  await expect.poll(tooltipVisible).toBe(false);
  expect(await readChart(chart, (instance) => instance.getOption().tooltip[0].formatter([{ dataIndex: 27 }]))).toBe('');
  await page.screenshot({ path: testInfo.outputPath(`gap-${width}.png`) });
  const gapBounds = await status.boundingBox();
  const groupBounds = await group.boundingBox();
  expect(gapBounds.y).toBeGreaterThanOrEqual(groupBounds.y + groupBounds.height - 1);
  expect(await status.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
  await page.keyboard.press('ArrowLeft');
  await expect(status).toContainText('qa-keyboard-0');
  await page.keyboard.press('ArrowRight');
  await expect(status).toContainText('qa-keyboard-1');
  await page.keyboard.press('Home');
  await expect(status).toContainText('No published run in this view');
  const firstStatus = await status.innerText();
  await page.keyboard.press('ArrowLeft');
  await expect(status).toHaveText(firstStatus);
  await page.keyboard.press('Escape');
  await expect(status).toBeEmpty();
  await expect(group).toBeFocused();
  expect(await cards.innerText()).toBe(originalCards);
  expect(await changes.innerText()).toBe(originalChanges);
  expect(await downloadSource(page)).toEqual(publication);

  // Existing nearest-slot pointer contract must still refuse the fully missing column.
  const pixel = await readChart(chart, (instance) => ({ x: instance.convertToPixel({ xAxisIndex: 0 }, 27), width: instance.getWidth() }));
  const box = await chart.boundingBox();
  await page.mouse.move(box.x + pixel.x * box.width / pixel.width, box.y + box.height / 2);
  await expect(status).toBeEmpty();
  await expect(trend.getByTestId('point-hover-guide')).not.toBeVisible();

  for (const run of publication.runs.slice(1)) for (const result of run.benchmark_results) {
    Object.assign(result, { status: 'failed', timing_results_s: [], error: 'QA-only all-gap comparison' });
  }
  publication.runs[0].provenance.rocjitsu.rocjitsu_commit_timestamp = '2026-09-10T10:00:00Z';
  publication.runs[0].execution_summary.finished_at = '2026-09-10T12:00:00Z';
  publication.runs[0].execution_summary.timestamp = '2026-09-10T11:00:00Z';
  await page.reload(); await ready(page);
  await readChart(trend.getByRole('img'), (instance) => instance.getOption().series.length);
  await group.focus(); await page.keyboard.press('End');
  await expect(status).toContainText('qa-keyboard-3');
  await expect(status).toContainText('Failed workload');
  const wrapper = await group.elementHandle();
  const week = trend.getByRole('button', { name: 'Trailing 7 days' });
  await week.click();
  await expect(week).toBeFocused();
  expect(await wrapper.evaluate((element) => element.isConnected)).toBe(true);
  await expect(status).toContainText('Comparison scope updated');
  await expect(status).not.toContainText('qa-keyboard-3');
  await expect(trend.getByRole('img')).toHaveCount(0);
  await expect(trend.getByTestId('speedup-empty-state')).toHaveText('No complete runs for selected comparison in this week.');
  await expect(trend.getByTestId('speedup-empty-state')).toBeVisible();
  // Fractional desktop CSS zoom can round the resolved height by a subpixel.
  expect(await trend.getByTestId('speedup-empty-state').evaluate((element) => parseFloat(getComputedStyle(element).height))).toBeCloseTo(280, 0);
  await group.focus(); await page.keyboard.press('End');
  await expect(status).toContainText('No eligible plotted run');
  await expect(status).not.toContainText('qa-keyboard-3');
  await page.screenshot({ path: testInfo.outputPath(`all-gap-${width}.png`) });
  await expect(trend).not.toContainText('provisional');

  Object.assign(last, savedLast);
  publication.index.runFiles = [`runs/default-branch/${last.publication.id}.json`];
  publication.runs = [last];
  await page.goto('/?targets=gfx1250&modes=single&range=1M'); await ready(page);
  await readChart(trend.getByRole('img'), (instance) => instance.getOption().series.length);
  await group.focus(); await page.keyboard.press('Home');
  await expect(status).toContainText('No published run in this view');
  await page.keyboard.press('End');
  await expect(status).toContainText('qa-keyboard-3');
  await expect(status).toContainText('1.000×');
  await expect(page.getByTestId('metric-card-valid-data-points').locator('[data-configuration="gfx1250/single"] p').last()).toHaveText('1');
  expect(await downloadSource(page)).toEqual(publication);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});
