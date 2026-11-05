import { expect, test } from '@playwright/test';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { selectOverviewProjection } from '../../src/data/overviewProjection.js';
import { formatDuration } from '../../src/utils/formatters.js';
import { readChart } from './helpers/chart.js';
import { installPublication, openDashboard, ready } from './helpers/dashboard.js';

test('Overview shares endpoints with cards and keeps hover/catalog/range/filter state coherent', async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  const publication = createMaintainedPublication();
  await installPublication(page, { publication });
  await openDashboard(page, '/?range=1W&campaign=kept#scope');
  const data = adaptPublishedDashboardData(publication).data;
  const model = selectOverviewProjection(data, { targets: data.targets, suites: data.suites, modes: data.modes }, { range: '1W' });
  expect(new URL(page.url()).searchParams.get('range')).toBe('1W');
  const cards = page.getByTestId('metric-strip');
  await expect(cards.locator('[data-testid^="metric-card-"]')).toHaveCount(4);
  expect(await cards.locator('[data-testid^="metric-card-"]').evaluateAll((elements) => elements.map((element) => element.dataset.testid)))
    .toEqual(['metric-card-latest-duration', 'metric-card-relative-speedup', 'metric-card-latest-results', 'metric-card-valid-data-points']);
  const changesPanel = page.getByTestId('largest-changes');
  const assertFlatChanges = async (projection) => {
    const expected = projection.rows.flatMap((row) => {
      if (!(row.latest?.durationSeconds > 0 && row.reference?.durationSeconds > 0)) return [];
      return row.latest.results.flatMap(({ id, result }) => {
        if (projection.catalogId == null && !['a', 'b'].includes(id)) return [];
        const baseline = row.reference.results.find((item) => item.id === id)?.result;
        const delta = baseline?.durationSeconds > 0 ? (result.durationSeconds - baseline.durationSeconds) / baseline.durationSeconds * 100 : null;
        return Number.isFinite(delta) ? [{ id, delta, key: row.key, target: row.target, mode: row.mode, suite: result.suite }] : [];
      });
    }).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta) || a.id.localeCompare(b.id)).slice(0, 6);
    const entries = changesPanel.locator('[data-change-state]');
    await expect(entries).toHaveCount(expected.length);
    for (const [index, item] of expected.entries()) {
      await expect(entries.nth(index)).toHaveAttribute('data-configuration', item.key);
      await expect(entries.nth(index)).toHaveAttribute('data-workload', item.id);
      for (const label of [item.target, item.mode, item.suite]) await expect(entries.nth(index)).toContainText(label);
    }
    await expect(changesPanel.locator('[data-configuration-group], h3')).toHaveCount(0);
    await expect(changesPanel.getByRole('combobox')).toHaveCount(0);
  };
  await assertFlatChanges(model);
  for (const row of model.rows) {
    await expect(page.getByTestId('metric-card-latest-duration').locator(`[data-configuration="${row.key}"]`)).toContainText(formatDuration(row.latest.durationSeconds));
    await expect(page.getByTestId('metric-card-relative-speedup').locator(`[data-configuration="${row.key}"]`)).toContainText(`${row.latest.speedup.toFixed(2)}×`);
    await expect(page.getByTestId('metric-card-latest-results').locator(`[data-configuration="${row.key}"]`)).toContainText(`${row.latest.completed}/${row.latest.required}`);
    await expect(page.getByTestId('metric-card-valid-data-points').locator(`[data-configuration="${row.key}"]`).locator('p').last()).toHaveText(String(row.points.filter((point) => Number.isFinite(point.speedup)).length));
  }
  await page.screenshot({ path: `${process.env.TMPDIR}/top-cards-populated.png`, fullPage: false });
  const trend = page.getByTestId('performance-trend');
  const chart = trend.getByRole('img');
  const plotted = await readChart(chart, (instance) => instance.getOption().series.map((s) => ({ name: s.name, connectNulls: s.connectNulls, showSymbol: s.showSymbol, area: s.areaStyle })));
  expect(plotted.map((s) => s.name)).toEqual(model.rows.map((r) => `${r.target} ${r.mode}`));
  expect(plotted.every((s) => s.connectNulls === false && s.showSymbol === false && !s.area)).toBe(true);
  const before = await cards.innerText();
  const group = trend.getByRole('group', { name: 'Inspect performance trend', exact: true });
  await group.press('End');
  await expect(trend.getByRole('status')).toContainText(model.latestRun.runId);
  expect(await cards.innerText()).toBe(before);
  await trend.getByRole('button', { name: 'Trailing 30 days' }).click();
  await expect(trend.getByTestId('point-hover-guide')).not.toBeVisible();
  await expect(trend.getByTestId('hover-line-focus')).not.toBeVisible();
  await trend.getByRole('button', { name: 'Trailing 7 days' }).click();
  await group.press('Home'); await group.press('ArrowRight'); await group.press('Escape');
  await expect(trend.getByRole('status')).toHaveCount(0);
  const pixel = await readChart(chart, (instance) => {
    const point = instance.getOption().series[0].data.findLast((p) => Number.isFinite(p[1]));
    return { pixel: instance.convertToPixel({ seriesIndex: 0 }, point), width: instance.getWidth(), height: instance.getHeight() };
  });
  const box = await chart.boundingBox();
  await page.mouse.move(box.x + pixel.pixel[0] * box.width / pixel.width, box.y + pixel.pixel[1] * box.height / pixel.height);
  await expect(trend.getByTestId('point-hover-guide')).toBeVisible();
  expect(await cards.innerText()).toBe(before);
  await page.mouse.move(0, 0);
  await expect(trend.getByTestId('point-hover-guide')).not.toBeVisible();
  await trend.locator('summary').click();
  await expect(trend.getByRole('columnheader').nth(1)).toHaveText('Shared across benchmark sets');
  await trend.getByRole('button', { name: 'Trailing 30 days' }).click();
  await trend.getByRole('combobox', { name: 'Benchmark set', exact: true }).click();
  await page.getByRole('option', { name: /fictional-old/ }).click();
  const catalogId = new URL(page.url()).searchParams.get('catalog');
  expect(catalogId).toBeTruthy();
  await page.reload(); await ready(page);
  expect(new URL(page.url()).searchParams.get('catalog')).toBe(catalogId);
  expect(new URL(page.url()).searchParams.get('campaign')).toBe('kept');
  expect(new URL(page.url()).hash).toBe('#scope');
  await page.getByRole('checkbox', { name: 'default', exact: true }).uncheck();
  await expect(page.getByTestId('metric-card-latest-duration').locator('[data-configuration]')).toHaveCount(2);
  await page.getByRole('checkbox', { name: 'Llama', exact: true }).uncheck();
  const filtered = selectOverviewProjection(data, { targets: data.targets, modes: ['single'], suites: ['Triton'] }, { range: '1M', catalogId });
  for (const row of filtered.rows) {
    await expect(page.getByTestId('metric-card-latest-results').locator(`[data-configuration="${row.key}"]`).locator('p').last()).toHaveText(`${row.latest.completed}/${row.latest.required}`);
    await expect(page.getByTestId('metric-card-valid-data-points').locator(`[data-configuration="${row.key}"]`).locator('p').last()).toHaveText(String(row.points.filter((point) => Number.isFinite(point.speedup)).length));
  }
  await assertFlatChanges(filtered);
  await page.getByRole('checkbox', { name: 'gfx950', exact: true }).uncheck();
  await assertFlatChanges(selectOverviewProjection(data, { targets: ['gfx1250'], modes: ['single'], suites: ['Triton'] }, { range: '1M', catalogId }));
  await page.getByRole('checkbox', { name: 'single', exact: true }).uncheck();
  await expect(changesPanel).toContainText('No configurations selected');
  await expect(changesPanel.locator('[data-change-state]')).toHaveCount(0);
  await page.screenshot({ path: `${process.env.TMPDIR}/c2-desktop.png`, fullPage: false });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: `${process.env.TMPDIR}/c2-narrow.png`, fullPage: false });
  expect(errors).toEqual([]);
});

test('Largest Changes Shared excludes a larger nonshared delta and catalog switching restores eligible entries', async ({ page }) => {
  const publication = createMaintainedPublication();
  const latest = publication.runs.find(({ publication: { id } }) => id === 'fictional-develop-23');
  const added = latest.benchmark_results.find(({ target, thread_policy, id }) => target === 'gfx1250' && thread_policy === 'single' && id === 'd');
  added.timing_results_s = [2000];
  await installPublication(page, { publication });
  await openDashboard(page, '/?targets=gfx1250&modes=single&range=1W');
  const panel = page.getByTestId('largest-changes');
  const shared = async () => {
    await expect(panel.locator('[data-change-state]')).toHaveCount(2);
    for (const id of ['a', 'b']) await expect(panel.locator(`[data-workload="${id}"]`)).toHaveCount(1);
    for (const id of ['c', 'd', 'e']) await expect(panel.locator(`[data-workload="${id}"]`)).toHaveCount(0);
  };
  await shared();
  const catalog = page.getByRole('combobox', { name: 'Benchmark set', exact: true });
  await catalog.click();
  await page.locator('[role="option"][data-value="fictional-current"]').click();
  await expect(panel.locator('[data-change-state]')).toHaveCount(4);
  await expect(panel.locator('[data-change-state]').first()).toHaveAttribute('data-workload', 'd');
  await catalog.click();
  await page.getByRole('option', { name: 'Shared benchmarks', exact: true }).click();
  await shared();
  await page.getByRole('button', { name: 'Trailing 30 days' }).click();
  await shared();
  await catalog.click();
  await page.locator('[role="option"][data-value="fictional-old"]').click();
  await expect(panel.locator('[data-workload="c"]')).toHaveCount(1);
  await catalog.click();
  await page.getByRole('option', { name: 'Shared benchmarks', exact: true }).click();
  await shared();
});

test('explicit empty configuration and suite selections remain empty through reload', async ({ page }) => {
  await openDashboard(page, '/?targets=&modes=&suites=&range=1M&campaign=kept#scope');
  await expect(page.getByTestId('metric-card-latest-duration')).toContainText('No configurations selected');
  await expect(page.getByTestId('speedup-empty-state')).toBeVisible();
  await expect(page.getByRole('img')).toHaveCount(0);
  await page.getByRole('button', { name: 'Reload all data' }).click(); await ready(page);
  await page.reload(); await ready(page);
  for (const key of ['targets', 'modes', 'suites']) expect(new URL(page.url()).searchParams.getAll(key)).toEqual(['']);
  expect(new URL(page.url()).searchParams.get('campaign')).toBe('kept');
  expect(new URL(page.url()).hash).toBe('#scope');
  await expect(page.getByTestId('speedup-empty-state')).toBeVisible();
});
