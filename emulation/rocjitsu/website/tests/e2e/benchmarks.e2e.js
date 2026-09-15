import { expect, test } from '@playwright/test';
import {
  clickAggregateIncompletePoint,
  clickCompletedChartPoint,
  clickLastCompletedChartPoint,
  clickLastStatusChartPoint,
  dispatchZoom,
  readChart,
  selectedDotsViewport,
  selectedRunCount,
  settledChartScale,
} from './helpers/chart.js';
import { expectDialogTypographyContained } from './helpers/dialog.js';

test('benchmark picker can locally reveal suites hidden by global filters', async ({ page }) => {
  await page.goto('/');

  await page.getByLabel('Suites').click();
  await page.getByRole('option', { name: /Check all suites/ }).click();
  await page.getByRole('option', { name: 'Triton' }).click();
  await page.keyboard.press('Escape');

  await page.getByRole('tab', { name: 'Benchmarks' }).click();
  await expect(page.getByText('Benchmark Explorer')).toBeVisible();
  await page.getByRole('combobox', { name: 'Benchmark' }).click();
  await expect(page.getByText('3 benchmarks hidden by global Suite filters')).toBeVisible();
  await page.getByRole('button', { name: 'Show all benchmarks' }).click();
  await page.getByRole('combobox', { name: 'Benchmark' }).fill('DeepSeek');
  await page.getByRole('option', { name: /DeepSeek V3 FP8 decode/ }).click();

  await expect(page.getByText('Outside global Suite filter')).toBeVisible();
  await expect(page.getByText(/This local override applies only to Benchmark Explorer/)).toBeVisible();
  await page.getByLabel('Suites').click();
  await expect(page.getByRole('option', { name: 'Triton' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('option', { name: 'DeepSeek' })).toHaveAttribute('aria-selected', 'false');
});

test('benchmark explorer switches among single, grid, and aggregate modes', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Benchmarks' }).click();

  const singleMode = page.getByRole('button', { name: 'Single' });
  const clickDetails = page.getByRole('button', { name: 'Disable details on click' });
  const scrollZoom = page.getByRole('button', { name: 'Disable scroll zoom' });
  await expect(singleMode).toHaveAttribute('aria-pressed', 'true');
  const controlPositions = await Promise.all([singleMode, clickDetails, scrollZoom].map(async (control) => {
    const bounds = await control.boundingBox();
    return { top: bounds.y, center: bounds.x + bounds.width / 2 };
  }));
  expect(controlPositions[1].top).toBeGreaterThan(controlPositions[0].top);
  expect(Math.abs(controlPositions[1].top - controlPositions[2].top)).toBeLessThanOrEqual(1);
  const benchmarkInputBox = await page.getByRole('combobox', { name: 'Benchmark' }).locator('..').boundingBox();
  const detailsButtonBox = await clickDetails.boundingBox();
  expect(Math.abs(
    (benchmarkInputBox.y + benchmarkInputBox.height / 2)
    - (detailsButtonBox.y + detailsButtonBox.height / 2),
  )).toBeLessThanOrEqual(1);
  await expect(page.getByRole('img', { name: /duration history/ })).toHaveCount(1);

  await page.getByRole('button', { name: 'Grid' }).click();
  await expect(page.getByText('1 of 8 benchmarks selected', { exact: false })).toBeVisible();
  await page.getByRole('combobox', { name: 'Benchmarks to graph' }).click();
  const option = page.getByRole('option', { name: /GEMM BF16 4096/ });
  await expect(option.getByRole('checkbox')).not.toBeChecked();
  await option.click();
  await expect(option.getByRole('checkbox')).toBeChecked();
  await page.keyboard.press('Escape');

  await expect(page.getByText('2 of 8 benchmarks selected', { exact: false })).toBeVisible();
  await expect(page.getByTestId('benchmark-grid').getByRole('img', { name: /duration history/ })).toHaveCount(2);
  await page.getByRole('button', { name: 'Hide GEMM BF16 4096³ graph' }).click();
  await expect(page.getByTestId('benchmark-grid').getByRole('img', { name: /duration history/ })).toHaveCount(1);

  await page.getByRole('button', { name: 'Aggregate' }).click();
  await expect(page.getByRole('button', { name: 'Aggregate' })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('Selected-suite duration by target across all official attempts, including reruns')).toBeVisible();
  const aggregateInstructions = page.getByText(/official attempts · Selected runs remain visible while zooming/);
  const aggregateInstructionBox = await aggregateInstructions.boundingBox();
  const aggregateDetailsBox = await page.getByRole('button', { name: 'Disable details on click' }).boundingBox();
  expect(aggregateInstructionBox.y + aggregateInstructionBox.height).toBeLessThanOrEqual(aggregateDetailsBox.y);

  const aggregateChart = page.getByRole('img', { name: 'Aggregate duration history for all runs' });
  await expect(aggregateChart).toBeVisible();
  const aggregateOption = await readChart(aggregateChart, (instance) => {
    const option = instance.getOption();
    return {
      runCount: option.xAxis[0].data.length,
      lineSeries: option.series.filter((series) => (
        series.type === 'line' && !series.name.includes('missed-data bridge')
      )).map((series) => ({
        name: series.name,
        showSymbol: series.showSymbol,
        hasLatestMarker: Boolean(series.markPoint?.data?.length),
      })),
      gapBridgeTypes: option.series
        .filter((series) => series.name.includes('missed-data bridge'))
        .map((series) => series.lineStyle.type),
      yAxisName: option.yAxis[0].name,
      yAxisScale: option.yAxis[0].scale,
      legendTargets: option.legend[0].data,
      zoomTypes: option.dataZoom.map((item) => item.type),
      initialStart: option.dataZoom[0].startValue,
      initialEnd: option.dataZoom[0].endValue,
    };
  });
  expect(aggregateOption.lineSeries.map((series) => series.name)).toEqual(['gfx1250']);
  expect(aggregateOption.lineSeries.every((series) => series.showSymbol === false)).toBe(true);
  expect(aggregateOption.lineSeries.every((series) => series.hasLatestMarker === false)).toBe(true);
  expect(aggregateOption.gapBridgeTypes.length).toBeGreaterThan(0);
  expect(aggregateOption.gapBridgeTypes.every((type) => type === 'dotted')).toBe(true);
  expect(aggregateOption.yAxisName).toBe('Seconds');
  expect(aggregateOption.yAxisScale).toBe(true);
  expect(aggregateOption.legendTargets).toEqual(aggregateOption.lineSeries.map((series) => series.name));
  expect(aggregateOption.zoomTypes.sort()).toEqual(['inside', 'slider']);
  expect(aggregateOption.initialEnd - aggregateOption.initialStart + 1).toBe(45);
  expect(aggregateOption.runCount).toBeGreaterThan(aggregateOption.initialEnd - aggregateOption.initialStart);

  await dispatchZoom(aggregateChart, 82, 100);
  const scaleBeforeSelection = await settledChartScale(aggregateChart);
  const disableDetailsOnClick = page.getByRole('button', { name: 'Disable details on click' });
  await expect(disableDetailsOnClick).toHaveText('Click details · On');
  await disableDetailsOnClick.click();
  const enableDetailsOnClick = page.getByRole('button', { name: 'Enable details on click' });
  await expect(enableDetailsOnClick).toHaveAttribute('aria-pressed', 'false');
  await expect(enableDetailsOnClick).toHaveText('Click details · Off');

  await clickLastCompletedChartPoint(aggregateChart);
  await expect(page.getByText('Selected 31369c4d · Auto')).toBeVisible();
  await expect(page.getByRole('dialog')).toHaveCount(0);
  const clearRuns = page.getByRole('button', { name: /Clear selected runs/ });
  await expect(clearRuns).toHaveText('Clear selected runs (1)');
  expect(await settledChartScale(aggregateChart)).toEqual(scaleBeforeSelection);

  await clickLastCompletedChartPoint(aggregateChart);
  await expect(clearRuns).toHaveText('Clear selected runs (0)');
  expect(await settledChartScale(aggregateChart)).toEqual(scaleBeforeSelection);

  await clickLastCompletedChartPoint(aggregateChart);
  await expect(clearRuns).toHaveText('Clear selected runs (1)');
  await clickCompletedChartPoint(aggregateChart, 1);
  await expect(clearRuns).toHaveText('Clear selected runs (2)');
  await expect(page.getByText('2 selected runs · latest selection 9f774d29')).toBeVisible();

  await dispatchZoom(aggregateChart, 99.9, 100);
  await expect.poll(async () => {
    const viewport = await selectedDotsViewport(aggregateChart, 'Selected run');
    return viewport.indexes.length === 2 && viewport.allVisible;
  }).toBe(true);
  const aggregateSelectedViewport = await selectedDotsViewport(aggregateChart, 'Selected run');
  expect(aggregateSelectedViewport.startValue).toBeLessThanOrEqual(Math.min(...aggregateSelectedViewport.indexes));
  expect(aggregateSelectedViewport.endValue).toBeGreaterThanOrEqual(Math.max(...aggregateSelectedViewport.indexes));

  await page.getByRole('button', { name: 'Single' }).click();
  await expect(page.getByRole('button', { name: 'Single' })).toHaveAttribute('aria-pressed', 'true');
  const singleChart = page.getByRole('img', { name: /duration history/ });
  expect(await selectedRunCount(singleChart)).toBe(2);
  await expect.poll(async () => (await selectedDotsViewport(singleChart, 'selected points')).allVisible).toBe(true);

  await page.getByRole('button', { name: 'Grid' }).click();
  const gridChart = page.getByTestId('benchmark-grid').getByRole('img', { name: /duration history/ }).first();
  expect(await selectedRunCount(gridChart)).toBe(2);
  await expect.poll(async () => (await selectedDotsViewport(gridChart, 'selected points')).allVisible).toBe(true);

  await page.getByRole('button', { name: 'Aggregate' }).click();
  await expect(page.getByText('2 selected runs · latest selection 9f774d29')).toBeVisible();
  await expect.poll(() => readChart(aggregateChart, (instance) => {
    const selected = instance.getOption()?.series?.find((candidate) => candidate.name === 'Selected run');
    return new Set((selected?.data ?? []).filter(Boolean).map((point) => point.run.runId)).size;
  })).toBe(2);

  await enableDetailsOnClick.click();
  await expect(page.getByRole('button', { name: 'Disable details on click' })).toHaveText('Click details · On');
  await clickLastCompletedChartPoint(aggregateChart);
  const runDialog = page.getByRole('dialog');
  await expect(runDialog.getByText('Run Details')).toBeVisible();
  await expect(runDialog.getByText('Commit 31369c4d · Auto')).toBeVisible();
  const commitLabel = await runDialog.getByText('RocJitsu commit', { exact: true }).boundingBox();
  const messageLabel = await runDialog.getByText('Commit message', { exact: true }).boundingBox();
  expect(messageLabel.x).toBeGreaterThan(commitLabel.x);
  await expectDialogTypographyContained(runDialog);
  await page.getByRole('button', { name: 'Close Run Details' }).click();
});

test('benchmark explorer scroll zoom is enabled by default and can be disabled', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Benchmarks' }).click();

  const chart = page.getByRole('img', { name: /duration history/ });
  const disableScrollZoom = page.getByRole('button', { name: 'Disable scroll zoom' });
  await expect(disableScrollZoom).toHaveAttribute('aria-pressed', 'true');
  await expect(disableScrollZoom).toHaveText('Scroll zoom · On');

  const insideZoomDisabled = () => readChart(chart, (instance) => instance.getOption().dataZoom[0].disabled);
  await expect.poll(insideZoomDisabled).toBe(false);

  await disableScrollZoom.click();
  const enableScrollZoom = page.getByRole('button', { name: 'Enable scroll zoom' });
  await expect(enableScrollZoom).toHaveAttribute('aria-pressed', 'false');
  await expect(enableScrollZoom).toHaveText('Scroll zoom · Off');
  await expect.poll(insideZoomDisabled).toBe(true);
});

test('benchmark history bridges gaps and opens failed result details', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Benchmarks' }).click();

  const chart = page.getByRole('img', { name: 'GEMM FP16 1024³ duration history' });
  await expect(chart).toBeVisible();
  const gapPresentation = await readChart(chart, (instance) => {
    const option = instance.getOption();
    return {
      bridgeTypes: option.series
        .filter((series) => series.name.includes('missed-data bridge'))
        .map((series) => series.lineStyle.type),
      statuses: option.series
        .find((series) => series.name === 'Failed or timed-out test results')
        .data.map((point) => point.record.test.status),
    };
  });
  expect(gapPresentation.bridgeTypes.length).toBeGreaterThan(0);
  expect(gapPresentation.bridgeTypes.every((type) => type === 'dotted')).toBe(true);
  expect(gapPresentation.statuses).toContain('failed');

  await clickLastStatusChartPoint(chart);
  const dialog = page.getByRole('dialog');
  await expect(dialog.getByText('Failed', { exact: true })).toBeVisible();
  await expect(dialog.getByText('Simulation exited before producing a valid timing result')).toBeVisible();
  await dialog.getByRole('button', { name: 'Close details' }).click();

  const failedSelection = await readChart(chart, (instance) => {
    const option = instance.getOption();
    const marker = option.series
      .find((series) => series.name === 'Failed or timed-out test results')
      ?.data.find((point) => point.selected);
    return {
      markerSize: marker?.symbolSize,
      borderWidth: marker?.itemStyle?.borderWidth,
      shadowBlur: marker?.itemStyle?.shadowBlur,
      hasRippleEffect: option.series.some((series) => series.type === 'effectScatter'),
    };
  });
  expect(failedSelection).toMatchObject({ borderWidth: 4, shadowBlur: 13, hasRippleEffect: false });
  expect(failedSelection.markerSize).toBeGreaterThanOrEqual(14);
  expect(failedSelection.markerSize).toBeLessThanOrEqual(17);
});

test('an incomplete recent run has clickable aggregate and failed-test markers', async ({ page }) => {
  await page.goto('/');

  const incompleteRun = page.getByTestId('recent-runs-table').locator('tbody tr').filter({ hasText: '19872076' });
  await incompleteRun.getByRole('button', { name: 'View run 19872076 in Benchmark Explorer' }).click();

  await expect(page.getByRole('tab', { name: 'Benchmarks' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: 'Aggregate' })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('Selected 19872076 · Auto')).toBeVisible();

  const aggregateChart = page.getByRole('img', { name: 'Aggregate duration history for all runs' });
  await expect(aggregateChart).toBeVisible();
  const selection = await readChart(aggregateChart, (instance) => {
    const option = instance.getOption();
    const incompletePoint = option.series
      .find((series) => series.name === 'Incomplete aggregate results')
      ?.data.find((point) => point?.run?.provenance?.rocjitsuCommitSha?.startsWith('19872076'));
    return {
      yAxisName: option.yAxis[0].name,
      target: incompletePoint?.target,
      completed: incompletePoint?.completed,
      total: incompletePoint?.total,
      label: incompletePoint?.label?.formatter,
      symbolSize: incompletePoint?.symbolSize,
      borderWidth: incompletePoint?.itemStyle?.borderWidth,
      shadowBlur: incompletePoint?.itemStyle?.shadowBlur,
    };
  });
  expect(selection).toMatchObject({
    yAxisName: 'Seconds',
    target: 'gfx1250',
    completed: 5,
    total: 7,
    label: '5/7',
    symbolSize: 17,
    borderWidth: 4,
    shadowBlur: 13,
  });

  await clickAggregateIncompletePoint(aggregateChart, '19872076');
  const runDetails = page.getByRole('dialog', { name: 'Run Details' });
  await expect(runDetails.getByRole('heading', { name: 'Selected Scope' })).toBeVisible();
  await expect(runDetails.getByText('5/7 completed')).toBeVisible();
  await expect(runDetails.getByText('Incomplete Tests')).toBeVisible();
  await expect(runDetails.getByText('GEMM FP16 1024³')).toBeVisible();
  await expect(runDetails.getByText('Failed', { exact: true })).toBeVisible();
  await expect(runDetails.getByText('Simulation exited before producing a valid timing result')).toBeVisible();
  await expect(runDetails.getByText('Softmax FP32 4096×4096')).toBeVisible();
  await expect(runDetails.getByText('Timeout', { exact: true })).toBeVisible();
  await expect(runDetails.getByText('Benchmark exceeded its configured timeout')).toBeVisible();
  await runDetails.getByRole('button', { name: 'Close Run Details' }).click();

  await page.getByRole('button', { name: 'Single' }).click();
  const failedSelection = await readChart(
    page.getByRole('img', { name: /duration history/ }),
    (instance) => {
      const point = instance.getOption().series
        .find((series) => series.name === 'Failed or timed-out test results')
        ?.data.find((candidate) => candidate.record?.run?.provenance?.rocjitsuCommitSha?.startsWith('19872076'));
      return {
        sha: point?.record?.run?.provenance?.rocjitsuCommitSha?.slice(0, 8),
        label: point?.label?.formatter,
        status: point?.record?.test?.status,
      };
    },
  );
  expect(failedSelection).toEqual({ sha: '19872076', label: 'Failed', status: 'failed' });
});

test('historical rows and completed chart points open shared provenance details', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Benchmarks' }).click();

  await page.getByTestId('historical-records')
    .getByRole('row', { name: /31369c4d/ })
    .getByRole('button', { name: /^Open result details/ })
    .click();
  const dialog = page.getByRole('dialog');
  await expect(dialog.getByText('Environment', { exact: true })).toBeVisible();
  await expect(dialog.getByText('Run Provenance')).toBeVisible();
  expect(await dialog.getByText(/^(Environment|Run Provenance)$/).allTextContents()).toEqual([
    'Environment',
    'Run Provenance',
  ]);
  await expect(dialog.getByRole('link', { name: 'Commit 31369c4d' })).toBeVisible();
  await expect(dialog.getByText('Validate follow-up scheduler tuning')).toBeVisible();
  await expect(dialog.getByText('Status', { exact: true })).toBeVisible();
  await expect(dialog.getByText('Duration', { exact: true })).toBeVisible();
  await expect(dialog.getByText('Operation', { exact: true }).locator('..')).toContainText('GEMM');
  await expect(dialog.getByText('Data type', { exact: true }).locator('..')).toContainText('fp16');
  await expect(dialog.getByText('ROCm SDK', { exact: true })).toBeVisible();
  await expect(dialog.getByText('7.2.0.dev202608', { exact: true })).toBeVisible();
  await expect(dialog.getByText('PyTorch', { exact: true })).toBeVisible();
  await expect(dialog.getByText('Raw problem configuration', { exact: true })).toHaveCount(0);
  const machineLabel = await dialog.getByText('Machine', { exact: true }).boundingBox();
  const commitLabel = await dialog.getByText('RocJitsu commit', { exact: true }).boundingBox();
  expect(commitLabel.x).toBeGreaterThan(machineLabel.x);
  await expectDialogTypographyContained(dialog);
  await page.getByRole('button', { name: 'Close details' }).click();

  const chart = page.getByRole('img', { name: 'GEMM FP16 1024³ duration history' });
  const clearRuns = page.getByRole('button', { name: /Clear selected runs/ });
  const disableDetails = page.getByRole('button', { name: 'Disable details on click' });
  await expect(clearRuns).toHaveText('Clear selected runs (1)');
  await clearRuns.click();
  await expect(clearRuns).toBeDisabled();
  await disableDetails.click();

  const chartPresentation = await readChart(chart, (instance) => {
    const option = instance.getOption();
    const zoom = option.dataZoom[0];
    const visibleValues = option.series
      .filter((series) => series.type === 'line')
      .flatMap((series) => series.data
        .slice(zoom.startValue, zoom.endValue + 1)
        .map((point) => point?.value)
        .filter(Number.isFinite));
    return {
      lineSymbols: option.series.filter((series) => series.type === 'line').map((series) => series.showSymbol),
      xAxisName: option.xAxis[0].name,
      yAxisName: option.yAxis[0].name,
      yAxisMinimum: option.yAxis[0].min,
      yAxisMaximum: option.yAxis[0].max,
      visibleMinimum: Math.min(...visibleValues),
      visibleMaximum: Math.max(...visibleValues),
      tooltipRenderMode: option.tooltip[0].renderMode,
      tooltipTriggerOn: option.tooltip[0].triggerOn,
    };
  });
  expect(chartPresentation.lineSymbols.every((showSymbol) => showSymbol === false)).toBe(true);
  expect(chartPresentation.xAxisName).toBe('Commit date / commit');
  expect(chartPresentation.yAxisName).toBe('Seconds');
  expect(chartPresentation.yAxisMinimum).toBeGreaterThan(0);
  expect(chartPresentation.yAxisMaximum - chartPresentation.yAxisMinimum).toBeGreaterThanOrEqual(
    (chartPresentation.visibleMaximum - chartPresentation.visibleMinimum) * 2,
  );
  expect(chartPresentation.tooltipRenderMode).not.toBe('richText');
  expect(chartPresentation.tooltipTriggerOn).toBe('mousemove');

  await clickLastCompletedChartPoint(chart);
  await expect(dialog).not.toBeVisible();
  await expect(clearRuns).toHaveText('Clear selected runs (1)');
  await clickLastCompletedChartPoint(chart);
  await expect(clearRuns).toHaveText('Clear selected runs (0)');

  await page.getByRole('button', { name: 'Enable details on click' }).click();
  await clickLastCompletedChartPoint(chart);
  await expect(dialog.getByRole('link', { name: 'Commit 31369c4d' })).toBeVisible();
  await page.getByRole('button', { name: 'Close details' }).click();
  await expect(clearRuns).toHaveText('Clear selected runs (1)');

  // Keep enough horizontal distance from the selected latest point, because adjacent scatter hit
  // targets intentionally overlap.
  await clickCompletedChartPoint(chart, 5);
  await expect(dialog).toBeVisible();
  await page.getByRole('button', { name: 'Close details' }).click();
  await expect(clearRuns).toHaveText('Clear selected runs (2)');
  await expect.poll(() => selectedRunCount(chart)).toBe(2);
  await clearRuns.click();
  await expect(clearRuns).toBeDisabled();
});

test('opens result and run details with the keyboard alone and pins the run like a click', async ({ page }) => {
  await page.goto('/');

  const resultAction = page.getByTestId('latest-results')
    .getByRole('button', { name: /^Open GEMM FP16 1024³ result details/ })
    .first();
  await resultAction.focus();
  await expect(resultAction).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('dialog').getByRole('heading', { name: 'Result' })).toBeVisible();
  await page.keyboard.press('Escape');

  await page.getByRole('tab', { name: 'Benchmarks' }).click();
  await expect(page.getByRole('button', { name: /^Clear selected runs \(0\)/ })).toBeVisible();
  await expect(page.getByRole('img', { name: 'GEMM FP16 1024³ duration history' }))
    .toHaveAttribute('aria-describedby', 'benchmark-chart-keyboard-help-triton-gemm-f16-1024');
  await expect(page.getByRole('combobox', { name: 'GEMM FP16 1024³ result' })).toBeVisible();
  const historyAction = page.getByTestId('historical-records')
    .getByRole('button', { name: /^Open result details/ })
    .first();
  await historyAction.focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('dialog').getByRole('heading', { name: 'Result' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: /^Clear selected runs \(1\)/ })).toBeVisible();

  await page.getByRole('button', { name: 'Grid' }).click();
  await expect(page.getByRole('img', { name: 'GEMM FP16 1024³ duration history' }))
    .toHaveAttribute('aria-describedby', 'benchmark-chart-keyboard-help-triton-gemm-f16-1024');
  await page.getByRole('combobox', { name: 'GEMM FP16 1024³ result' }).focus();
  await page.keyboard.press('ArrowDown');
  await expect(page.getByRole('option').first()).toBeVisible();
  await page.keyboard.press('Enter');
  await page.getByRole('button', { name: 'Open GEMM FP16 1024³ result' }).press('Enter');
  await expect(page.getByRole('dialog').getByRole('heading', { name: 'Result' })).toBeVisible();
  await page.keyboard.press('Escape');

  await page.getByRole('button', { name: 'Aggregate' }).click();
  await expect(page.getByRole('img', { name: 'Aggregate duration history for all runs' }))
    .toHaveAttribute('aria-describedby', 'aggregate-chart-keyboard-help');
  await page.getByRole('combobox', { name: 'Aggregate run' }).focus();
  await page.keyboard.press('ArrowDown');
  await expect(page.getByRole('option').first()).toBeVisible();
  await page.keyboard.press('Enter');
  await page.getByRole('button', { name: 'Open run details' }).press('Enter');
  await expect(page.getByRole('dialog').getByRole('heading', { name: 'Run Details' })).toBeVisible();
});

test('the chart point selector searches every run but renders a bounded option list', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Benchmarks' }).click();
  await page.getByRole('button', { name: 'Aggregate' }).click();

  const runSelector = page.getByRole('combobox', { name: 'Aggregate run' });
  await expect(runSelector).not.toHaveValue('');
  await runSelector.click();
  const options = page.getByRole('option');
  await expect(options).toHaveCount(50);
  const totalMatches = Number(
    (await page.getByText(/Search all \d+ runs/).textContent()).match(/all (\d+)/)[1],
  );
  expect(totalMatches).toBeGreaterThan(50);

  await runSelector.fill('zzzznomatch');
  await expect(options).toHaveCount(0);
  await expect(page.getByText('No options')).toBeVisible();

  // A non-default run outside the rendered window is reachable with the keyboard.
  await runSelector.fill('9f774d29');
  await expect(options).toHaveCount(1);
  await page.keyboard.press('Enter');
  await page.getByRole('button', { name: 'Open run details' }).click();
  await expect(page.getByRole('dialog').getByText('Commit 9f774d29 · Auto')).toBeVisible();
});

test('historical records renders one bounded page at a time', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Benchmarks' }).click();

  const history = page.getByTestId('historical-records');
  await expect(history.locator('tbody tr')).toHaveCount(25);
  await expect(history.getByText(/^1–25 of \d+ results$/)).toBeVisible();
  await history.getByRole('button', { name: 'Go to next page' }).click();
  await expect(history.getByText(/^26–50 of \d+ results$/)).toBeVisible();
  await expect(history.locator('tbody tr')).toHaveCount(25);

  await history.getByRole('combobox', { name: 'Rows per page' }).click();
  await page.getByRole('option', { name: '50' }).click();
  await expect(history.getByText(/^1–50 of \d+ results$/)).toBeVisible();
  await expect(history.locator('tbody tr')).toHaveCount(50);
});
