import { expect, test } from '@playwright/test';
import { readChart } from './helpers/chart.js';

test('metric captions use the card width beneath their icons', async ({ page }) => {
  await page.goto('/');

  const cards = page.getByTestId(/^metric-card-/);
  await expect(cards).toHaveCount(4);

  for (let index = 0; index < 4; index += 1) {
    const card = cards.nth(index);
    const [cardBox, iconBox, captionBox] = await Promise.all([
      card.boundingBox(),
      card.getByTestId('metric-icon').boundingBox(),
      card.getByTestId('metric-caption').boundingBox(),
    ]);

    expect(captionBox.x).toBeGreaterThan(cardBox.x + 12);
    expect(captionBox.x + captionBox.width).toBeGreaterThanOrEqual(iconBox.x + iconBox.width);
    expect(captionBox.x + captionBox.width).toBeLessThan(cardBox.x + cardBox.width - 12);
  }
});

test('performance trend fills the row beside largest changes', async ({ page }) => {
  await page.goto('/');

  const trend = page.getByTestId('performance-trend');
  const changes = page.getByTestId('largest-changes');
  const chartWrapper = page.getByTestId('performance-trend-chart');
  const chart = trend.getByRole('img', { name: 'Performance trend for ALL' });
  await expect(trend.getByText('Range change', { exact: true })).toBeVisible();
  await expect(trend.getByText('Latest selected total', { exact: true })).toBeVisible();
  await expect(trend.getByTestId('performance-trend-normalization-note')).toContainText(
    'Performance trend values are normalized to the latest test catalog',
  );
  const normalizationNote = trend.getByTestId('performance-trend-normalization-note');
  const noteText = normalizationNote.locator('.MuiAlert-message > span');
  const [noteBox, textBox] = await Promise.all([
    normalizationNote.boundingBox(),
    noteText.boundingBox(),
  ]);
  expect(noteBox).not.toBeNull();
  expect(textBox).not.toBeNull();
  expect(Math.abs(
    (textBox.y + textBox.height / 2) - (noteBox.y + noteBox.height / 2),
  )).toBeLessThanOrEqual(1);
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
      yAxisScale: option.yAxis[0].scale,
      series: option.series
        .filter((series) => series.type === 'line')
        .map((series) => ({
          showSymbol: series.showSymbol,
          connectNulls: series.connectNulls,
          smooth: series.smooth,
          latestMarkers: series.markPoint?.data?.length ?? 0,
          baselineLabel: series.markLine?.label?.formatter,
        })),
    };
  });
  expect(markers.yAxisScale).toBe(true);
  expect(markers.series.every((series) => series.showSymbol === false)).toBe(true);
  expect(markers.series.every((series) => series.connectNulls === true)).toBe(true);
  expect(markers.series.every((series) => Number(series.smooth) > 0)).toBe(true);
  expect(markers.series.every((series) => series.latestMarkers === 1)).toBe(true);
  expect(markers.series.every((series) => !series.baselineLabel?.includes('est.'))).toBe(true);
});

test('opens the Benchmarks tab from the normalization note', async ({ page }) => {
  await page.goto('/');

  await page.getByTestId('performance-trend-normalization-note')
    .getByRole('button', { name: 'Benchmarks' })
    .click();

  await expect(page.getByRole('tab', { name: 'Benchmarks' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByText('Benchmark Explorer')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Aggregate' })).toHaveAttribute('aria-pressed', 'true');
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
  await page.getByRole('heading', { name: 'Rocjitsu Simulation Performance' }).click();

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
  const august30 = recentRuns.getByRole('row').filter({
    has: page.getByRole('button', { name: 'Compare 9398bd3f' }),
  });
  const comparison = august30.getByTestId('compared-commits');
  await expect(comparison).toBeVisible();
  const initialComparison = await comparison.getAttribute('aria-label');

  await page.getByLabel('Suites').click();
  await page.getByRole('option', { name: 'TensileLite' }).click();
  await page.getByRole('option', { name: 'DeepSeek' }).click();
  await page.keyboard.press('Escape');

  await expect(comparison).not.toHaveAttribute('aria-label', initialComparison);
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
  const labels = ['Target', 'Suite', 'Benchmark', 'Duration', 'Baseline', 'Delta abs', 'Delta %', 'Status'];
  for (const label of labels) {
    const header = results.getByRole('columnheader', { name: label, exact: true });
    await header.getByRole('button', { name: label, exact: true }).click();
    await expect(header).toHaveAttribute('aria-sort', 'ascending');
  }

  const deltaPercent = results.locator('tbody tr td:nth-child(7) [data-change-state]').first();
  await expect(deltaPercent).toBeVisible();
  await expect(results.locator('[data-change-state="slower"]')).toHaveCount(0);
  const fasterColor = await results.locator('[data-change-state="faster"]').first().evaluate((el) => (
    getComputedStyle(el).color
  ));
  const unavailableColor = await results.locator('[data-change-state="unavailable"]').first().evaluate((el) => (
    getComputedStyle(el).color
  ));
  expect(fasterColor).toBe('rgb(22, 138, 91)');
  expect(unavailableColor).toBe('rgb(100, 112, 135)');

  const durationHeader = results.getByRole('columnheader', { name: 'Duration', exact: true });
  await expect(durationHeader.getByRole('button', { name: 'Duration', exact: true })).toHaveCSS('flex-direction', 'row');
  await expect(durationHeader.locator('.MuiTableSortLabel-icon')).toHaveCSS('position', 'absolute');
  const durationCell = results.locator('tbody tr').first().locator('td').nth(3);
  const rightPadding = await Promise.all([
    durationHeader.evaluate((element) => getComputedStyle(element).paddingRight),
    durationCell.evaluate((element) => getComputedStyle(element).paddingRight),
  ]);
  expect(rightPadding[0]).toBe(rightPadding[1]);
  await durationHeader.getByRole('button', { name: 'Duration', exact: true }).click();
  const durations = (await results.locator('tbody tr td:nth-child(4)').allTextContents()).map((value) => {
    const minutes = Number(value.match(/(\d+)m/)?.[1] ?? 0);
    const seconds = Number(value.match(/([\d.]+)s/)?.[1] ?? 0);
    return minutes * 60 + seconds;
  });
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

test('global filters split the full width evenly and collapse excess labels into +N', async ({ page }) => {
  await page.goto('/');

  const targets = page.getByTestId('targets-filter');
  const suites = page.getByTestId('suites-filter');
  const filterLayout = await targets.evaluate((element) => {
    const grid = element.parentElement.getBoundingClientRect();
    const targetBounds = element.getBoundingClientRect();
    const suiteBounds = element.parentElement.querySelector('[data-testid="suites-filter"]')
      .getBoundingClientRect();
    return {
      gridLeft: grid.left,
      gridRight: grid.right,
      targetLeft: targetBounds.left,
      targetWidth: targetBounds.width,
      suiteRight: suiteBounds.right,
      suiteWidth: suiteBounds.width,
    };
  });
  expect(Math.abs(filterLayout.targetWidth - filterLayout.suiteWidth)).toBeLessThanOrEqual(1);
  expect(Math.abs(filterLayout.targetLeft - filterLayout.gridLeft)).toBeLessThanOrEqual(1);
  expect(Math.abs(filterLayout.suiteRight - filterLayout.gridRight)).toBeLessThanOrEqual(1);
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
  await expect(page.getByText(/\d+ commits? shown/)).toBeVisible();
  await expect(page.getByTestId('performance-range-change')).toHaveAttribute('data-change-state', 'faster');
  await expect(page.getByText(/Latest vs first shown/)).toHaveCount(0);
  const largestChangesPair = page.getByTestId('largest-changes').getByTestId('compared-commits').first();
  const latestResultsPair = page.getByTestId('latest-results').getByTestId('compared-commits').first();
  const allRangePair = await largestChangesPair.getAttribute('aria-label');
  expect(allRangePair).toBe('Candidate commit 31369c4d versus baseline commit 86b362ea');
  expect(await latestResultsPair.getAttribute('aria-label')).toBe(allRangePair);

  await page.getByRole('button', { name: 'Trailing 7 days' }).click();
  await expect(page.getByText(/\d+ commits? shown/)).toBeVisible();
  await expect(page.getByText('past week', { exact: true })).toBeVisible();
  await expect(page.getByTestId('performance-range-change')).toHaveAttribute('data-change-state', 'faster');
  const weekRangePair = await largestChangesPair.getAttribute('aria-label');
  expect(weekRangePair).not.toBe(allRangePair);
  expect(await latestResultsPair.getAttribute('aria-label')).toBe(weekRangePair);
  const weekAxis = await readChart(
    page.getByRole('img', { name: 'Performance trend for 1W' }),
    (instance) => ({
      name: instance.getOption().xAxis[0].name,
      labels: instance.getModel().getComponent('xAxis').axis
        .getViewLabels()
        .map((label) => label.formattedLabel)
        .filter(Boolean),
    }),
  );
  expect(weekAxis.name).toBe('Date (UTC)');
  expect(weekAxis.labels).toHaveLength(7);
  expect(new Set(weekAxis.labels).size).toBe(7);

  await page.getByRole('button', { name: 'Trailing 24 hours' }).click();
  await expect(page.getByText('3 commits shown')).toBeVisible();
  await expect(page.getByText('past day', { exact: true })).toBeVisible();
  await expect(page.getByTestId('performance-range-change')).toHaveAttribute('data-change-state', 'slower');
  const dayRangePair = await largestChangesPair.getAttribute('aria-label');
  expect(dayRangePair).not.toBe(weekRangePair);
  expect(await latestResultsPair.getAttribute('aria-label')).toBe(dayRangePair);
  const intradayTooltip = await readChart(
    page.getByRole('img', { name: 'Performance trend for 1D' }),
    (instance) => {
      const option = instance.getOption();
      const series = option.series.find((candidate) => candidate.type === 'line');
      const dataIndex = series.data.findIndex((value) => (
        Array.isArray(value) && Number.isFinite(value[1])
      ));
      return option.tooltip[0].formatter([{
        dataIndex,
        value: series.data[dataIndex],
        marker: '',
        seriesName: series.name,
      }]);
    },
  );
  expect(intradayTooltip).toContain('Commit date ·');
  expect(intradayTooltip).toMatch(/Commit SHA · [a-f0-9]{8}/);
  expect(intradayTooltip).toContain('Test catalog ·');
  expect(intradayTooltip).not.toContain('Run time ·');
  expect(intradayTooltip).not.toContain('Commit time ·');
  expect(intradayTooltip).toMatch(/\d+m\d+s/);
  expect(intradayTooltip).toMatch(/Time change [+-]?\d+\.\d+%/);
  expect(intradayTooltip).toMatch(/Time change [+-]?\d+\.\d+% · Base time \d+m\d+s/);
  expect(intradayTooltip).not.toMatch(/<strong>Base time/);
  expect(intradayTooltip).not.toMatch(/\d+m \d+/);
  const durationUnit = await readChart(
    page.getByRole('img', { name: 'Performance trend for 1D' }),
    (instance) => instance.getOption().yAxis[0].name,
  );
  expect(durationUnit).toBe('Minutes');
  const intradayAxis = await readChart(
    page.getByRole('img', { name: 'Performance trend for 1D' }),
    (instance) => {
      const option = instance.getOption();
      const baseline = option.series[0].markLine.data[0].yAxis;
      return {
        name: option.xAxis[0].name,
        type: option.xAxis[0].type,
        points: option.series[0].data.length,
        baseline,
        yAxisMin: option.yAxis[0].min,
        yAxisMax: option.yAxis[0].max,
      };
    },
  );
  expect(intradayAxis.name).toBe('Commit time (UTC)');
  expect(intradayAxis.type).toBe('value');
  expect(intradayAxis.points).toBe(3);
  expect(intradayAxis.yAxisMin).toBeLessThanOrEqual(intradayAxis.baseline);
  expect(intradayAxis.yAxisMax).toBeGreaterThanOrEqual(intradayAxis.baseline);
});

test('keeps a dashed baseline above the measured range inside the y-axis', async ({ page }) => {
  await page.route('**/data/runs/generated-commit-20260830T153000000Z.json', async (route) => {
    const response = await route.fetch();
    const run = await response.json();
    run.targets.forEach((target) => target.results.forEach((result) => {
      if (Number.isFinite(result.durationSeconds)) result.durationSeconds *= 10;
    }));
    await route.fulfill({ response, json: run });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Trailing 24 hours' }).click();

  const bounds = await readChart(
    page.getByRole('img', { name: 'Performance trend for 1D' }),
    (instance) => {
      const option = instance.getOption();
      const baselines = option.series.map((series) => series.markLine.data[0].yAxis);
      const measuredValues = option.series.flatMap((series) => (
        series.data.map((point) => point[1]).filter(Number.isFinite)
      ));
      return {
        highestBaseline: Math.max(...baselines),
        highestMeasuredValue: Math.max(...measuredValues),
        yAxisMax: option.yAxis[0].max,
      };
    },
  );

  expect(bounds.highestBaseline).toBeGreaterThan(bounds.highestMeasuredValue);
  expect(bounds.yAxisMax).toBeGreaterThanOrEqual(bounds.highestBaseline);
});

test('labels the 1D axis once per commit shown', async ({ page }) => {
  for (const file of ['benchmark-202608310530-255eabe3', 'benchmark-202608311310-9f774d29']) {
    await page.route(`**/data/runs/${file}.json`, async (route) => {
      const response = await route.fetch();
      const run = await response.json();
      run.targets.forEach((target) => target.results.forEach((result) => {
        result.durationSeconds = null;
        result.status = 'failed';
        result.error = 'Simulation exited before producing a valid timing result';
      }));
      await route.fulfill({ response, json: run });
    });
  }
  await page.goto('/');
  await page.getByRole('button', { name: 'Trailing 24 hours' }).click();
  await expect(page.getByText('1 commit shown')).toBeVisible();

  const intradayLabels = await readChart(
    page.getByRole('img', { name: 'Performance trend for 1D' }),
    (instance) => instance.getModel().getComponent('xAxis').axis
      .getViewLabels()
      .map((label) => label.formattedLabel)
      .filter(Boolean),
  );
  expect(intradayLabels).toEqual(['19:45']);
});

test('uses a continuous value x-axis for every rendered trend range', async ({ page }) => {
  await page.goto('/');

  for (const rangeButton of ['All available history', 'Trailing 30 days', 'Trailing 7 days', 'Trailing 24 hours']) {
    await page.getByRole('button', { name: rangeButton }).click();
    const axisType = await readChart(
      page.getByRole('img', { name: /Performance trend for/ }),
      (instance) => instance.getOption().xAxis[0].type,
    );
    expect(axisType, rangeButton).toBe('value');
    const axisBounds = await readChart(
      page.getByRole('img', { name: /Performance trend for/ }),
      (instance) => {
        const option = instance.getOption();
        const baselineValues = option.series
          .map((series) => series.markLine?.data?.[0]?.yAxis)
          .filter((value) => Number.isFinite(value));
        const values = option.series.flatMap((series) => (
          series.data
            .filter((value) => Array.isArray(value) && Number.isFinite(value[1]))
            .map((value) => value[1])
        ));
        return {
          min: option.yAxis[0].min,
          max: option.yAxis[0].max,
          baselineCount: baselineValues.length,
          lowest: Math.min(...values, ...baselineValues),
          highest: Math.max(...values),
        };
      },
    );
    expect(axisBounds.baselineCount, rangeButton).toBeGreaterThan(0);
    const decimalPlaces = Math.max(
      0,
      Math.ceil(-Math.log10((axisBounds.highest - axisBounds.lowest) / 5)),
    );
    const precision = 10 ** decimalPlaces;
    expect(axisBounds.min, rangeButton).toBeCloseTo(
      Math.floor(axisBounds.lowest * precision) / precision,
      5,
    );
    expect(axisBounds.max, rangeButton).toBeCloseTo(
      Math.ceil(axisBounds.highest * precision) / precision,
      5,
    );
  }
});

test('shows the minimum 3M coverage and insufficient-data messaging for sparser ranges', async ({ page }) => {
  await page.goto('/');

  await page.getByRole('button', { name: 'Trailing 90 days' }).click();
  await expect(page.getByTestId('performance-trend-chart')).toBeVisible();
  await expect(page.getByTestId('performance-trend-insufficient')).toHaveCount(0);

  await page.getByRole('button', { name: 'Trailing 180 days' }).click();
  await expect(page.getByTestId('performance-trend-insufficient')).toHaveText(
    "There isn't enough data for the selected time range",
  );
});

test('largest changes uses a fixed three-percent noise tolerance', async ({ page }) => {
  await page.goto('/');

  const largestChanges = page.getByTestId('largest-changes');
  await expect(page.getByLabel('Noise tolerance')).toHaveCount(0);
  await expect(page.getByText('Largest benchmark changes across the selected history range', { exact: true })).toBeVisible();
  await expect(largestChanges.locator('[data-legend-state="slower"]')).toHaveText('Slower');
  await expect(largestChanges.locator('[data-legend-state="neutral"]')).toHaveText('Within ±3%');
  await expect(largestChanges.locator('[data-legend-state="faster"]')).toHaveText('Faster');
  await expect(largestChanges.getByText(/Slower >|Faster </)).toHaveCount(0);
  await expect(largestChanges.locator('[data-change-state="neutral"]')).toHaveCount(0);
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
