import { expect, test } from '@playwright/test';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
import { formatDuration } from '../../src/utils/formatters.js';
import { readChart } from './helpers/chart.js';
import { openDashboard, ready } from './helpers/dashboard.js';

test('all-page shell, theme and overview controls retain develop-only selected-scope semantics', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  await openDashboard(page);
  const reload = page.getByRole('button', { name: 'Reload all data', exact: true });
  for (let step = 0; step < 14; step++) {
    await page.keyboard.press('Tab');
    if (await reload.evaluate((element) => element === document.activeElement)) break;
  }
  await expect(reload).toBeFocused();
  await expect.poll(() => reload.evaluate((element) => {
    const style = getComputedStyle(element);
    return { style: style.outlineStyle, width: style.outlineWidth, offset: style.outlineOffset };
  })).toEqual({ style: 'solid', width: '2px', offset: '2px' });
  await expect(page.getByTestId('fixture-data-warning')).toContainText('fictional measurements');
  await expect(page.getByTestId('dashboard-navigation').getByRole('tab')).toHaveText([
    'Overview', 'Branch Runs', 'Benchmarks', 'Run Comparison',
  ]);
  await expect(page.getByRole('tab', { name: /Plugin|Failures/ })).toHaveCount(0);
  await expect(page.getByText('Beta', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('link', { name: 'Repository', exact: true }))
    .toHaveAttribute('href', 'https://github.com/ROCm/rocm-systems');
  const rows = page.getByTestId('recent-runs-table').locator('tbody tr');
  await expect(rows).toHaveCount(20);
  expect(await rows.evaluateAll((entries) => entries.map((entry) => entry.dataset.runId)))
    .toEqual(createSchema2Publication().runs.filter((run) => run.source.branch === 'develop' && run.plugin.id === 'vanilla')
      .sort((a, b) => Date.parse(b.execution.completedAt) - Date.parse(a.execution.completedAt) || b.id.localeCompare(a.id))
      .slice(0, 20).map((run) => run.id));
  await expect(rows.first()).toHaveAttribute('data-run-id', 'fictional-develop-21');
  await expect(rows.first().locator('time').first()).toHaveAttribute('datetime', '2026-10-05T10:00:00.000Z');
  await expect(rows.first().locator('time').nth(1)).toHaveAttribute('datetime', '2026-10-02T04:00:00.000Z');

  // One line is a selected-scope sum, not one line per target or synthetic MT.
  await page.getByRole('checkbox', { name: 'MT', exact: true }).uncheck();
  const latest = createSchema2Publication().runs.find((run) => run.id === 'fictional-develop-23');
  const sum = latest.configurations.find((config) => config.target === 'gfx1250' && config.mode === 'ST')
    .results.reduce((total, result) => total + result.durationSeconds, 0);
  await expect(page.getByTestId('metric-card-total-duration')).toContainText(formatDuration(sum));
  const trend = page.getByTestId('performance-trend');
  const plotted = await readChart(trend.getByRole('img'), (instance) => {
    const option = instance.getOption();
    return { names: option.series.map((series) => series.name), unit: option.yAxis[0].name,
      latest: option.series[0].data.filter((point) => Number.isFinite(point[1])).at(-1)[1] };
  });
  expect(plotted.names).toEqual(['Selected runtime']);
  expect(plotted.latest * (plotted.unit === 'Minutes' ? 60 : 1)).toBeCloseTo(sum, 8);
  const inspection = trend.getByRole('group', { name: 'Inspect performance trend' });
  await inspection.press('Home');
  await expect(trend.getByRole('status')).toContainText(createSchema2Publication().runs.find((run) => run.id === 'fictional-develop-00').source.commit.slice(0, 8));
  await trend.getByText(/^Inspect first-success anchors/).click();
  await expect(trend.getByRole('region', { name: 'Selected trend estimate anchors' }))
    .toContainText('first success fictional-develop-09');
  await expect(trend.getByRole('region', { name: 'Selected trend estimate anchors' }))
    .toContainText('first success fictional-develop-10');
  await inspection.press('End');
  await expect(trend.getByRole('status')).toContainText(latest.source.commit.slice(0, 8));
  await inspection.press('Escape');
  await expect(trend.getByRole('status')).toHaveCount(0);
  const chart = inspection.getByRole('img');
  const lastPixel = await readChart(chart, (instance) => {
    const point = instance.getOption().series[0].data.findLast((entry) => Number.isFinite(entry[1]));
    return instance.convertToPixel({ seriesIndex: 0 }, point);
  });
  const box = await chart.boundingBox();
  await page.mouse.move(box.x + lastPixel[0], box.y + lastPixel[1]);
  await expect(trend.getByRole('status')).toContainText(latest.source.commit.slice(0, 8));
  // Stable hover must settle, not feed repeated axis events back into chart renders.
  await readChart(chart, (instance) => instance.getOption().series[0].data.length);
  await page.mouse.move(0, 0);
  // Leaving hides transient guides, not persistent inspection. Escape clears it.
  await expect(trend.getByRole('status')).toContainText(latest.source.commit.slice(0, 8));
  await inspection.press('Escape');
  await expect(trend.getByRole('status')).toHaveCount(0);
  for (const [label, range] of [['Trailing 7 days', '1W'], ['Trailing 30 days', '1M'], ['Trailing 90 days', '3M'], ['All available history', 'ALL']]) {
    await trend.getByRole('button', { name: label }).click();
    await expect(trend.getByRole('button', { name: label })).toHaveAttribute('aria-pressed', 'true');
    await expect.poll(() => new URL(page.url()).searchParams.get('range')).toBe(range);
  }
  await trend.getByRole('button', { name: 'Benchmarks', exact: true }).click();
  await expect(page.getByTestId('benchmark-grid')).toBeVisible();
  for (const name of ['Run Comparison', 'Branch Runs', 'Overview']) {
    await page.getByRole('tab', { name, exact: true }).press('Enter');
    await expect(page.getByRole('tabpanel', { name, exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name, exact: true, level: 1 })).toBeVisible();
  }
  await page.getByRole('button', { name: 'Use dark theme' }).click();
  await expect(page.getByRole('button', { name: 'Use light theme' })).toBeVisible();
  await page.reload();
  await ready(page);
  await expect(page.getByRole('button', { name: 'Use light theme' })).toBeVisible();
  await page.getByRole('button', { name: 'Use light theme' }).click();
  expect(errors).toEqual([]);
});

test('explicit empty target, suite and mode scopes never refill across page or data reloads', async ({ page }) => {
  await openDashboard(page, '/?compareCandidate=fictional-develop-23&compareBaseline=fictional-develop-20&campaign=kept#scope');
  // Establish real workload choices before narrowing their available scope.
  await page.getByRole('tab', { name: 'Benchmarks', exact: true }).click();
  const headings = page.getByTestId('benchmark-grid-card').getByRole('heading');
  await expect(headings).toHaveCount(4);
  const retained = await headings.allTextContents();
  // Defaults are derived/read-only until a user explicitly applies a choice.
  await page.getByRole('button', { name: 'Add benchmarks', exact: true }).click();
  await page.getByRole('dialog', { name: 'Add benchmarks', exact: true })
    .getByRole('button', { name: 'Apply selection', exact: true }).click();
  await expect(headings).toHaveText(retained);
  await page.getByRole('tab', { name: 'Overview', exact: true }).click();
  await page.getByRole('checkbox', { name: 'ST', exact: true }).uncheck();
  await page.getByRole('checkbox', { name: 'MT', exact: true }).uncheck();
  await expect(page.getByTestId('metric-card-run-health')).toContainText('No selected results available');
  await expect(page.getByTestId('metric-card-total-duration')).toContainText('—');
  await page.getByRole('tab', { name: 'Benchmarks', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('No benchmarks in the selected targets, suites and execution modes');
  await expect(headings).toHaveText(retained);
  await expect(page.getByRole('img')).toHaveCount(0);
  await page.getByRole('button', { name: 'Reload all data' }).click();
  await ready(page);
  await expect(page.getByTestId('execution-modes-filter').locator('input:checked')).toHaveCount(0);
  await expect(headings).toHaveText(retained);
  await page.reload();
  await ready(page);
  await expect(page.getByTestId('execution-modes-filter').locator('input:checked')).toHaveCount(0);
  await expect(headings).toHaveText(retained);
  expect(new URL(page.url()).searchParams.getAll('modes')).toEqual(['']);
  expect(new URL(page.url()).searchParams.get('campaign')).toBe('kept');
  expect(new URL(page.url()).hash).toBe('#scope');
  await page.getByRole('checkbox', { name: 'ST', exact: true }).check();
  await page.getByRole('checkbox', { name: 'gfx1250', exact: true }).uncheck();
  await expect(page.getByRole('img')).toHaveCount(0);
  await page.getByRole('checkbox', { name: 'gfx1250', exact: true }).check();
  for (const suite of ['Triton', 'Llama']) await page.getByRole('checkbox', { name: suite, exact: true }).uncheck();
  await expect(page.getByRole('img')).toHaveCount(0);
  await page.getByRole('tab', { name: 'Run Comparison', exact: true }).click();
  await expect(page.getByTestId('comparison-metric-comparable').getByText('0', { exact: true })).toBeVisible();
  await expect(page.getByText('No completed benchmark results are comparable between these runs.')).toBeVisible();
  await page.getByRole('checkbox', { name: 'Triton', exact: true }).check();
  await expect(page.getByRole('img', { name: 'Performance change by benchmark comparison chart' })).toBeVisible();
});
