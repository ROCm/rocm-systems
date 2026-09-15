import { expect, test } from '@playwright/test';
import { readChart } from './helpers/chart.js';

test('performance trend fills the row beside largest changes', async ({ page }) => {
  await page.goto('/');

  const trend = page.getByTestId('performance-trend');
  const changes = page.getByTestId('largest-changes');
  const chartWrapper = page.getByTestId('performance-trend-chart');
  const chart = trend.getByRole('img', { name: 'Performance trend for ALL' });
  const boxes = await Promise.all([
    trend.boundingBox(),
    changes.boundingBox(),
    chartWrapper.boundingBox(),
    chart.boundingBox(),
  ]);

  expect(Math.abs(boxes[0].height - boxes[1].height)).toBeLessThanOrEqual(1);
  expect(Math.abs(boxes[2].height - boxes[3].height)).toBeLessThanOrEqual(1);
  expect(boxes[3].height).toBeGreaterThanOrEqual(278);

  const markers = await readChart(chart, (instance) => {
    const option = instance.getOption();
    return {
      xAxisName: option.xAxis[0].name,
      yAxisScale: option.yAxis[0].scale,
      series: option.series
        .filter((series) => series.type === 'line')
        .map((series) => ({
          showSymbol: series.showSymbol,
          connectNulls: series.connectNulls,
          smooth: series.smooth,
          latestMarkers: series.markPoint?.data?.length ?? 0,
        })),
    };
  });
  expect(markers.xAxisName).toBe('Commit Date (UTC)');
  expect(markers.yAxisScale).toBe(true);
  expect(markers.series.every((series) => series.showSymbol === false)).toBe(true);
  expect(markers.series.every((series) => series.connectNulls === true)).toBe(true);
  expect(markers.series.every((series) => Number(series.smooth) > 0)).toBe(true);
  expect(markers.series.every((series) => series.latestMarkers === 1)).toBe(true);
});

test('failures count stays fully inside the navigation bar', async ({ page }) => {
  await page.goto('/');
  const navigation = page.getByTestId('dashboard-navigation');
  const failuresTab = page.getByRole('tab', { name: /Failures/ });
  const badge = failuresTab.locator('.MuiBadge-badge');
  const [navigationBox, tabBox, badgeBox] = await Promise.all([
    navigation.boundingBox(),
    failuresTab.boundingBox(),
    badge.boundingBox(),
  ]);

  expect(badgeBox.y).toBeGreaterThanOrEqual(navigationBox.y);
  expect(badgeBox.y + badgeBox.height).toBeLessThanOrEqual(navigationBox.y + navigationBox.height);
  expect(badgeBox.x).toBeGreaterThanOrEqual(tabBox.x);
  expect(badgeBox.x + badgeBox.width).toBeLessThanOrEqual(tabBox.x + tabBox.width);
});

test('uses contrasting target series and engineering-tone comparison labels', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Targets').click();
  await page.getByRole('option', { name: /Check all targets/ }).click();
  await page.keyboard.press('Escape');
  await page.getByRole('heading', { name: 'RocJitsu Performance Health' }).click();

  const trend = page.getByRole('img', { name: 'Performance trend for ALL' });
  const targetSeriesColors = await readChart(trend, (instance) => instance.getOption().series
    .filter((series) => series.type === 'line')
    .map((series) => series.lineStyle.color.toLowerCase()));
  expect(targetSeriesColors).toEqual(['#2166c1', '#c25430']);

  await page.getByRole('tab', { name: 'Run Comparison' }).click();

  const chart = page.getByRole('img', { name: 'Performance change by benchmark comparison chart' });
  const encoding = await readChart(chart, (instance) => {
    const option = instance.getOption();
    const barData = option.series.find((series) => series.type === 'bar').data;
    const yAxis = option.yAxis[0];
    return {
      fillCount: [...new Set(barData.map((item) => item.itemStyle.color))].length,
      usesEngineeringTokens: yAxis.data.every((label) => label.includes('{target|') && label.includes('{benchmark|')),
      richStyleKeys: Object.keys(yAxis.axisLabel.rich),
      targetLabelColor: yAxis.axisLabel.rich.target.color,
      benchmarkLabelColor: yAxis.axisLabel.rich.benchmark.color,
      hasOutlinedBars: barData.some((item) => item.itemStyle.borderWidth > 0),
    };
  });
  expect(encoding.fillCount).toBeLessThanOrEqual(3);
  expect(encoding.usesEngineeringTokens).toBe(true);
  expect(encoding.richStyleKeys.sort()).toEqual(['benchmark', 'separator', 'target']);
  expect(encoding.targetLabelColor).not.toBe(encoding.benchmarkLabelColor);
  expect(encoding.hasOutlinedBars).toBe(false);
});

test('recent runs waits for two selections before opening compare', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByText('Each perf change uses the latest run from the nearest earlier commit that completed all currently selected tests.')).toBeVisible();
  await expect(page.getByText('To compare runs, select the candidate first and the baseline second.')).toBeVisible();

  const recentRuns = page.getByTestId('recent-runs-table');
  await expect(recentRuns.getByRole('columnheader', { name: 'Actions' })).toBeVisible();
  await expect(recentRuns.getByRole('columnheader', { name: 'Run time (UTC)' })).toBeVisible();
  await expect(recentRuns.getByRole('columnheader', { name: 'Commit time (UTC)' })).toBeVisible();
  await expect(recentRuns.getByRole('columnheader', { name: 'Trigger' })).toHaveCount(0);
  await expect(recentRuns.getByRole('columnheader', { name: 'Run type' })).toBeVisible();
  const contentGroupCenters = await recentRuns.locator('tbody tr').first().locator('[data-table-cell-group]').evaluateAll(
    (elements) => elements.map((element) => {
      const bounds = element.getBoundingClientRect();
      return bounds.top + bounds.height / 2;
    }),
  );
  expect(Math.max(...contentGroupCenters) - Math.min(...contentGroupCenters)).toBeLessThanOrEqual(1);
  const rowHeights = await recentRuns.locator('tbody tr').evaluateAll(
    (rows) => rows.slice(0, 3).map((row) => row.getBoundingClientRect().height),
  );
  expect(Math.abs(rowHeights[0] - rowHeights[1])).toBeLessThanOrEqual(1);
  expect(rowHeights[1] - rowHeights[2]).toBeGreaterThanOrEqual(24);

  await expect(page.getByLabel('Show runs')).toHaveText('8 runs');
  await expect(page.getByText('Push', { exact: true })).toHaveCount(0);
  await expect(page.getByText('Nightly', { exact: true })).toHaveCount(0);
  await page.getByLabel('Show runs').click();
  await page.getByRole('option', { name: '5 runs' }).click();
  const runSelectors = page.getByRole('button', { name: /^Compare [a-f0-9]+$/ });
  const exploreActions = page.getByRole('button', { name: /^View run [a-f0-9]+ in Benchmark Explorer$/ });
  await expect(runSelectors).toHaveCount(5);
  await expect(exploreActions).toHaveCount(5);
  await runSelectors.first().click();

  await expect(page.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByText('Candidate selected')).toBeVisible();
  await expect(page.getByText('Select another run as the baseline to open Run Comparison.')).toBeVisible();
  await expect(runSelectors.first()).toHaveAttribute('aria-pressed', 'true');
  await expect(runSelectors.first()).toHaveCSS('background-color', 'rgb(85, 102, 233)');
  await runSelectors.nth(2).click();

  await expect(page.getByRole('tab', { name: 'Run Comparison' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('combobox', { name: 'Candidate run' })).toHaveValue(/8418072e/);
  await expect(page.getByRole('combobox', { name: 'Baseline run' })).toHaveValue(/9f774d29/);
  await expect(page.getByText('Candidate vs baseline · Lower duration is faster')).toBeVisible();
});

test('recent run baselines respect the active global test scope', async ({ page }) => {
  await page.goto('/');

  const recentRuns = page.getByTestId('recent-runs-table');
  const august30 = recentRuns.getByRole('row').filter({ hasText: '9398bd3f' });
  await expect(august30.getByLabel('Candidate commit 9398bd3f versus baseline commit daaf4bae')).toBeVisible();

  await page.getByLabel('Suites').click();
  await page.getByRole('option', { name: 'TensileLite' }).click();
  await page.getByRole('option', { name: 'DeepSeek' }).click();
  await page.keyboard.press('Escape');

  await expect(august30.getByLabel('Candidate commit 9398bd3f versus baseline commit 0db03af1')).toBeVisible();
});

test('a manual rerun of an older commit is labelled without becoming the latest commit', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByTestId('latest-commit-run')).toContainText('31369c4d');
  const newestExecution = page.getByTestId('recent-runs-table').locator('tbody tr').first();
  await expect(newestExecution).toContainText('8418072e');
  await expect(newestExecution).toContainText('Sep 1 · 01:15');
  await expect(newestExecution).toContainText('Aug 15 · 05:30');
  await expect(newestExecution.getByLabel('Most recent run')).toBeVisible();
  await expect(newestExecution).toContainText('Most recent');
  await expect(newestExecution).toContainText('Manual');
  await expect(newestExecution.getByLabel('Historical rerun')).toBeVisible();
  await expect(newestExecution).toContainText('Historical rerun');
  await expect(newestExecution).not.toContainText('Reference');
  await expect(newestExecution.locator('[title*="nearest earlier commit: f25f5a48"]')).toHaveCount(1);

  const newestCommitRun = page.getByTestId('recent-runs-table').locator('tbody tr').filter({ hasText: '31369c4d' });
  await expect(newestCommitRun).toHaveCount(1);
  await expect(newestCommitRun.getByLabel('Latest commit')).toBeVisible();
  await expect(newestCommitRun).toContainText('Latest commit');
  await expect(newestCommitRun.locator('[title*="nearest earlier commit: 9f774d29"]')).toHaveCount(1);
});

test('latest results can be sorted by every column', async ({ page }) => {
  await page.goto('/');

  const results = page.getByTestId('latest-results');
  const labels = ['Target', 'Suite', 'Benchmark', 'Type', 'Problem', 'Duration', 'Baseline', 'Change', 'Status'];
  for (const label of labels) {
    const header = results.getByRole('columnheader', { name: label, exact: true });
    await header.getByRole('button', { name: label, exact: true }).click();
    await expect(header).toHaveAttribute('aria-sort', 'ascending');
  }

  const durationHeader = results.getByRole('columnheader', { name: 'Duration', exact: true });
  await expect(durationHeader.getByRole('button', { name: 'Duration', exact: true })).toHaveCSS('flex-direction', 'row');
  await expect(durationHeader.locator('.MuiTableSortLabel-icon')).toHaveCSS('position', 'absolute');
  const durationCell = results.locator('tbody tr').first().locator('td').nth(5);
  const rightPadding = await Promise.all([
    durationHeader.evaluate((element) => getComputedStyle(element).paddingRight),
    durationCell.evaluate((element) => getComputedStyle(element).paddingRight),
  ]);
  expect(rightPadding[0]).toBe(rightPadding[1]);
  await durationHeader.getByRole('button', { name: 'Duration', exact: true }).click();
  const durations = (await results.locator('tbody tr td:nth-child(6)').allTextContents()).map((value) => Number.parseFloat(value));
  expect(durations).toEqual([...durations].sort((left, right) => left - right));
  await durationHeader.getByRole('button', { name: 'Duration', exact: true }).click();
  await expect(durationHeader).toHaveAttribute('aria-sort', 'descending');
});

test('latest results precedes recent runs and renders one bounded page', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Targets').click();
  await page.getByRole('option', { name: /Check all targets/ }).click();
  await page.keyboard.press('Escape');

  const results = page.getByTestId('latest-results');
  const recentRunsTable = page.getByTestId('recent-runs-table');
  const resultsComeFirst = await results.evaluate((element, recentRuns) => (
    Boolean(element.compareDocumentPosition(recentRuns) & Node.DOCUMENT_POSITION_FOLLOWING)
  ), await recentRunsTable.elementHandle());
  expect(resultsComeFirst).toBe(true);

  await expect(results.locator('tbody tr')).toHaveCount(10);
  await expect(results.getByText('1–10 of 14 results')).toBeVisible();
  await results.getByRole('button', { name: 'Go to next page' }).click();
  await expect(results.getByText('11–14 of 14 results')).toBeVisible();
  await expect(results.locator('tbody tr')).toHaveCount(4);
});

test('target and suite filters use checkbox menus with check-all controls', async ({ page }) => {
  await page.goto('/');

  await page.getByLabel('Targets').click();
  await expect(page.getByRole('option', { name: /Check all targets/ })).toBeVisible();
  await expect(page.getByRole('option', { name: 'gfx1250' })).toBeVisible();
  await page.getByRole('option', { name: /Check all targets/ }).click();
  await expect(page.getByText('gfx950', { exact: true }).first()).toBeVisible();

  await page.keyboard.press('Escape');
  await page.getByLabel('Suites').click();
  await expect(page.getByRole('option', { name: /Check all suites/ })).toBeVisible();
  await expect(page.getByRole('option', { name: 'Triton' })).toBeVisible();
});

test('suite filter keeps labels visible until they approach the dropdown control', async ({ page }) => {
  await page.goto('/');

  const suites = page.getByTestId('suites-filter');
  await expect(suites.locator('[data-responsive-tag]')).toHaveCount(3);
  await expect(suites.locator('[data-overflow-tag]')).toHaveCount(0);

  await page.setViewportSize({ width: 600, height: 900 });
  await expect(suites.locator('[data-responsive-tag]')).toHaveCount(1);
  await expect(suites.locator('[data-overflow-tag]')).toHaveText('+2');
  const spacing = await suites.evaluate((element) => {
    const overflow = element.querySelector('[data-overflow-tag]').getBoundingClientRect();
    const controls = element.querySelector('.MuiAutocomplete-endAdornment').getBoundingClientRect();
    return controls.left - overflow.right;
  });
  expect(spacing).toBeGreaterThan(0);

  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(suites.locator('[data-responsive-tag]')).toHaveCount(3);
  await expect(suites.locator('[data-overflow-tag]')).toHaveCount(0);
});

test('keeps timeframe controls local to the duration history', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByLabel('Comparison period')).toHaveCount(0);
  await expect(page.getByLabel('History range')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'All available history' })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText(/\d+ UTC commit dates? shown/)).toBeVisible();
  await expect(page.getByTestId('performance-range-change')).toHaveAttribute('data-change-state', 'faster');
  await expect(page.getByTestId('performance-range-change')).toContainText('1.3%');
  await expect(page.getByText('Latest vs first shown in ALL')).toBeVisible();

  await page.getByRole('button', { name: 'Trailing 7 days' }).click();
  await expect(page.getByText(/\d+ UTC commit dates? shown/)).toBeVisible();
  await expect(page.getByTestId('performance-range-change')).toHaveAttribute('data-change-state', 'slower');
  await expect(page.getByTestId('performance-range-change')).toContainText('0.5%');
  await expect(page.getByText('Latest vs first shown in 1W')).toBeVisible();

  await page.getByRole('button', { name: 'Trailing 24 hours' }).click();
  await expect(page.getByText('3 completed runs shown')).toBeVisible();
  await expect(page.getByTestId('performance-range-change')).toHaveAttribute('data-change-state', 'faster');
  await expect(page.getByTestId('performance-range-change')).toContainText('1.2%');
  await expect(page.getByText('Latest vs first shown in 1D')).toBeVisible();
  const intradayTooltip = await readChart(
    page.getByRole('img', { name: 'Performance trend for 1D' }),
    (instance) => {
      const option = instance.getOption();
      const series = option.series.find((candidate) => candidate.type === 'line');
      const dataIndex = series.data.findIndex((value) => Number.isFinite(value));
      return option.tooltip[0].formatter([{
        dataIndex,
        value: series.data[dataIndex],
        marker: '',
        seriesName: series.name,
      }]);
    },
  );
  expect(intradayTooltip).toContain('Run time ·');
  expect(intradayTooltip).toContain('Commit time ·');
});

test('largest changes uses a fixed three-percent noise tolerance', async ({ page }) => {
  await page.goto('/');

  const largestChanges = page.getByTestId('largest-changes');
  await expect(page.getByLabel('Noise tolerance')).toHaveCount(0);
  await expect(page.getByText('Largest benchmark changes in the Overview comparison', { exact: true })).toBeVisible();
  await expect(largestChanges.locator('[data-legend-state="slower"]')).toHaveText('Slower');
  await expect(largestChanges.locator('[data-legend-state="neutral"]')).toHaveText('Within ±3%');
  await expect(largestChanges.locator('[data-legend-state="faster"]')).toHaveText('Faster');
  await expect(largestChanges.getByText(/Slower >|Faster </)).toHaveCount(0);
  await expect(largestChanges.locator('[data-change-state="neutral"]')).toHaveCount(3);
});

test('recent runs contains horizontal scrolling on compact laptops', async ({ page }) => {
  for (const width of [1024, 1280, 1440, 1920]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/');
    const documentWidths = await page.evaluate(() => ({
      client: document.documentElement.clientWidth,
      scroll: document.documentElement.scrollWidth,
    }));
    expect(documentWidths.scroll).toBeLessThanOrEqual(documentWidths.client);

    const table = page.getByTestId('recent-runs-table');
    await expect(table).toBeVisible();
    const tableWidths = await table.evaluate((element) => ({ client: element.clientWidth, scroll: element.scrollWidth }));
    if (width < 1200) {
      expect(tableWidths.scroll).toBeGreaterThan(tableWidths.client);
    } else {
      expect(tableWidths.scroll).toBeLessThanOrEqual(tableWidths.client);
    }
  }
});

test('opens benchmark details and toggles theme', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /^Open GEMM FP16 1024³ result details/ }).first().click();
  const dialog = page.getByRole('dialog');
  await expect(dialog.getByText('Result', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Close details' }).click();

  await page.getByRole('button', { name: 'Use dark theme' }).click();
  await expect(page.getByRole('button', { name: 'Use light theme' })).toBeVisible();
});
