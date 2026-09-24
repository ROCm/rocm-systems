import { expect, test } from '@playwright/test';
import { readChart } from './helpers/chart.js';

test('compares controlled plugins for every target without a target picker', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Plugin Comparison' }).click();

  const view = page.getByTestId('plugin-comparison');
  await expect(view.getByRole('heading', { name: 'Plugin Comparison' })).toBeVisible();
  await expect(view.getByRole('combobox', { name: 'Baseline plugin' })).toHaveText('Vanilla');
  await expect(page.getByLabel('Targets')).toBeVisible();
  const contract = view.getByText(/source commit/i);
  await expect(contract).toContainText(/catalog/i);
  await expect(contract).toContainText(/trigger/i);
  await expect(contract).toContainText(/machine/i);
  await expect(contract).toContainText(/environment/i);
  await expect(contract).toContainText(/target set/i);

  const gfx1250 = view.getByTestId('plugin-target-gfx1250');
  const gfx950 = view.getByTestId('plugin-target-gfx950');
  await expect(gfx1250.getByRole('heading', { name: 'gfx1250' })).toBeVisible();
  await expect(gfx950).toHaveCount(0);
  await page.getByLabel('Targets').click();
  await page.getByRole('option', { name: /Check all targets/ }).click();
  await page.keyboard.press('Escape');
  await expect(gfx950.getByRole('heading', { name: 'gfx950' })).toBeVisible();
  await expect(gfx1250.getByTestId('plugin-summary-gfx1250-vanilla')).toContainText('7/7');
  await expect(gfx1250.getByTestId('plugin-summary-gfx1250-asan')).toContainText('Geometric-mean runtime overhead');
  await expect(gfx1250.getByTestId('plugin-summary-gfx1250-tsan')).toContainText('6/7');
  await expect(gfx1250.getByTestId('plugin-summary-gfx1250-tsan')).toContainText(/\+\d+\.\d%\*/);
  await expect(gfx1250.getByTestId('plugin-summary-gfx1250-tsan')).toContainText('Estimated for all 7 tests from 6 passed pairs');
  await expect(gfx1250.getByTestId('plugin-summary-gfx1250-tsan')).not.toContainText('0 timed out');
  await expect(gfx950.getByTestId('plugin-summary-gfx950-tsan')).not.toContainText('0 failed');
  await expect(gfx1250.getByText(/geometric-mean overhead measured from passed plugin\/baseline pairs is assumed/)).toBeVisible();
  await expect(gfx1250.getByTestId('plugin-summary-gfx1250-ubsan')).toContainText('6/7');
  await expect(gfx1250.getByTestId('plugin-summary-gfx1250-ubsan')).toContainText('1 failed');
  await expect(gfx950.getByTestId('plugin-summary-gfx950-ubsan')).toContainText('1 timed out');
  const gfx1250Chart = gfx1250.getByRole('img', { name: 'Plugin runtime overhead for gfx1250' });
  await expect(gfx1250Chart).toBeVisible();
  await expect(gfx950.getByRole('img', { name: 'Plugin runtime overhead for gfx950' })).toBeVisible();
  await expect(gfx1250.getByText('Data race detected in scheduler state')).toBeVisible();

  const seriesColors = () => readChart(gfx1250Chart, (instance) => Object.fromEntries(
    instance.getOption().series.map((series) => [series.name, series.itemStyle.color]),
  ));
  expect(await seriesColors()).toEqual({
    AddressSanitizer: '#D97706',
    ThreadSanitizer: '#8B5CF6',
    UndefinedBehaviorSanitizer: '#0891B2',
  });

  await view.getByRole('combobox', { name: 'Baseline plugin' }).click();
  await page.getByRole('option', { name: 'AddressSanitizer' }).click();
  await expect(view.getByRole('combobox', { name: 'Baseline plugin' })).toHaveText('AddressSanitizer');
  await expect(gfx1250.getByTestId('plugin-summary-gfx1250-asan')).toContainText('Selected baseline');
  await expect(gfx1250.getByTestId('plugin-summary-gfx1250-asan')).not.toContainText('Uninstrumented reference');
  expect(await seriesColors()).toEqual({
    Vanilla: '#16A34A',
    ThreadSanitizer: '#8B5CF6',
    UndefinedBehaviorSanitizer: '#0891B2',
  });

  const layerNormRow = gfx1250.getByRole('row', { name: /LayerNorm BF16 8192/ });
  await layerNormRow.getByRole('button', { name: /Open ThreadSanitizer result/ }).click();
  const dialog = page.getByRole('dialog');
  await expect(dialog.getByText('ThreadSanitizer', { exact: true })).toBeVisible();
  await expect(dialog.getByText('LLVM 20', { exact: true })).toBeVisible();
  await expect(dialog.getByText('ThreadSanitizer: data race detected in scheduler state')).toBeVisible();
  await expect(dialog.getByText('Problem Details', { exact: true })).toBeVisible();
});

test('shows neutral copy when only baseline plugin data is published', async ({ page }) => {
  await page.route('**/data/index.json', async (route) => {
    const response = await route.fetch();
    const index = await response.json();
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ...index,
        runFiles: index.runFiles.filter((runFile) => !/-(?:asan|tsan|ubsan)\.json$/.test(runFile)),
      }),
    });
  });

  await page.goto('/');
  await page.getByRole('tab', { name: 'Plugin Comparison' }).click();

  const emptyState = page.getByTestId('plugin-comparison-empty');
  await expect(emptyState.getByText('No sanitizer comparison runs available')).toBeVisible();
  await expect(emptyState).toContainText('No vanilla baseline or sanitizer comparison runs are available');
  await expect(emptyState).toContainText('at least one sanitizer run for the same commit');
  await expect(emptyState).not.toHaveAttribute('role', 'alert');
  const view = page.getByTestId('plugin-comparison');
  await expect(view.getByRole('combobox', { name: 'Comparison experiment' })).toHaveText(/ · Vanilla$/);
  await expect(view.getByTestId('plugin-summary-gfx1250-vanilla')).toContainText('Vanilla');
  await expect(view.getByTestId('plugin-summary-gfx1250-vanilla')).toContainText('Baseline');
  await expect(view.getByText('Comparison plugin', { exact: true })).toBeVisible();
  await expect(view.getByText('Per-Test Runtime Overhead')).toBeVisible();
  await expect(view.getByRole('heading', { name: 'Test-by-Plugin Results' })).toBeVisible();
});

test('escapes injected catalog text instead of executing it in a chart tooltip', async ({ page }) => {
  const injectedName = '<img src=x onerror="window.__tooltipInjection = true">';
  await page.route('**/data/test-catalogs/*.json', async (route) => {
    const response = await route.fetch();
    const catalog = await response.json();
    catalog.tests.forEach((definition) => {
      if (definition.id === 'triton-gemm-f16-1024') definition.name = injectedName;
    });
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(catalog) });
  });

  const showInjectedTooltip = (chart) => readChart(chart, (instance) => {
    const option = instance.getOption();
    let located = null;
    option.series.forEach((series, seriesIndex) => series.data.forEach((point, dataIndex) => {
      const name = point?.test?.name ?? point?.comparison?.candidateTest?.name;
      if (!located && point?.value != null && name?.startsWith('<img')) {
        located = { seriesIndex, dataIndex, point };
      }
    }));
    if (!located) throw new Error('No chart point carries the injected benchmark name');
    instance.dispatchAction({ type: 'showTip', seriesIndex: located.seriesIndex, dataIndex: located.dataIndex });
    return option.tooltip[0].formatter({ data: located.point });
  });
  const expectEscaped = (tooltipHtml) => {
    expect(tooltipHtml).toContain('&lt;img src=x onerror=&quot;window.__tooltipInjection = true&quot;&gt;');
    expect(tooltipHtml).not.toContain('<img');
  };

  await page.goto('/');
  await page.getByRole('tab', { name: 'Plugin Comparison' }).click();
  const pluginChart = page.getByRole('img', { name: 'Plugin runtime overhead for gfx1250' });
  await expect(pluginChart).toBeVisible();
  expectEscaped(await showInjectedTooltip(pluginChart));

  await page.getByRole('tab', { name: 'Run Comparison' }).click();
  const compareChart = page.getByRole('img', { name: 'Performance change by benchmark comparison chart' });
  await expect(compareChart).toBeVisible();
  expectEscaped(await showInjectedTooltip(compareChart));

  expect(await page.evaluate(() => window.__tooltipInjection)).toBeUndefined();
});

test('comparison surfaces identify both compared commits', async ({ page }) => {
  await page.goto('/');

  const latestPair = 'Candidate commit 31369c4d versus baseline commit 9f774d29';
  const historyPair = 'Candidate commit 31369c4d versus baseline commit 86b362ea';
  await expect(page.getByTestId('performance-trend').getByLabel(historyPair)).toBeVisible();
  await expect(page.getByTestId('largest-changes').getByLabel(historyPair)).toHaveCount(5);
  await expect(page.getByTestId('latest-results').getByLabel(historyPair)).toHaveCount(5);

  const recentRuns = page.getByTestId('recent-runs-table');
  await expect(recentRuns.locator('tbody tr').first().getByLabel('Candidate commit 8418072e versus baseline commit f25f5a48')).toBeVisible();
  await expect(recentRuns.locator('tbody tr').filter({ hasText: '31369c4d' })
    .locator('[title*="nearest earlier commit: 9f774d29"]')).toHaveCount(1);

  // Automatic baselines are a comparison policy, not a user selection, so no dialog offers one.
  await page.getByTestId('latest-results')
    .getByRole('button', { name: /^Open GEMM FP16 1024³ result details/ })
    .first()
    .click();
  const overviewDialog = page.getByRole('dialog');
  await expect(overviewDialog.getByRole('heading', { name: 'Result' })).toBeVisible();
  await expect(overviewDialog.getByRole('heading', { name: 'Problem Details' })).toBeVisible();
  await expect(overviewDialog.getByRole('heading', { name: 'Run Provenance' })).toBeVisible();
  await expect(overviewDialog.getByText('Baseline run', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Close details' }).click();

  await page.getByRole('tab', { name: 'Benchmarks' }).click();
  const history = page.getByTestId('historical-records');
  const latestCommitRecord = history.locator('tbody tr').filter({ hasText: '31369c4d' });
  await expect(latestCommitRecord.getByLabel(latestPair)).toBeVisible();
  await expect(latestCommitRecord).toContainText('Validate follow-up scheduler tuning');
  await expect(latestCommitRecord.getByText('Validate follow-up scheduler tuning')).toHaveCSS('white-space', 'nowrap');

  // Result history compares against the previous completed result for the same test, which can be
  // an earlier commit than the run-level baseline.
  const august28 = history.getByRole('row', { name: /Aug 28, 2026/ });
  await expect(august28.getByLabel('Candidate commit 87c0b32c versus baseline commit 0db03af1')).toBeVisible();
  await expect(history.getByRole('row', { name: /Aug 27, 2026/ })
    .getByLabel('Candidate commit 0db03af1 versus baseline commit 390ca630')).toBeVisible();
  await august28.getByRole('button', { name: /^Open result details/ }).click();
  await expect(page.getByRole('dialog').getByText('Baseline run', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Close details' }).click();

  await page.getByRole('tab', { name: 'Run Comparison' }).click();
  await expect(page.getByText('Performance Change by Benchmark')).toBeVisible();
  await expect(page.getByText('Aggregate change')).toBeVisible();
  await expect(page.getByText('Not comparable', { exact: true })).toBeVisible();
  await expect(page.getByText('Only benchmarks with valid completed durations in both runs are compared.')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Compared Run Information' })).toBeVisible();
  const candidateInformation = page.getByTestId('candidate-run-information');
  const baselineInformation = page.getByTestId('baseline-run-information');
  await expect(candidateInformation).toContainText('8418072e');
  await expect(candidateInformation).toContainText('rocjitsu-core-v1');
  await expect(candidateInformation).toContainText('5/5 completed');
  await expect(candidateInformation).toContainText('ROCm SDK');
  await expect(baselineInformation).toContainText('31369c4d');
  await expect(baselineInformation).toContainText('rocjitsu-core-v2');
  await expect(baselineInformation).toContainText('7/7 completed');
  await expect(baselineInformation).toContainText('ROCm SDK');
  await expect(page.getByRole('heading', { name: 'Tests Not Included in Comparison' })).toBeVisible();
  await expect(page.getByRole('row', { name: /GEMM FP8 4096³.*Unavailable in catalog.*Completed/ })).toBeVisible();
  await expect(page.getByRole('row', { name: /DeepSeek V3 FP8 decode.*Unavailable in catalog.*Completed/ })).toBeVisible();
  await expect(page.getByLabel('Candidate commit 8418072e versus baseline commit 31369c4d')).toHaveCount(2);
});

test('run comparison search reaches a historical rerun through a bounded option list', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Run Comparison' }).click();

  const candidateRun = page.getByRole('combobox', { name: 'Candidate run' });
  await expect(candidateRun).toHaveValue(/8418072e/);
  await expect(page.getByRole('combobox', { name: 'Baseline run' })).toHaveValue(/31369c4d/);
  await candidateRun.click();
  await expect(page.getByRole('option')).toHaveCount(50);
  await candidateRun.fill('8418072e');
  const matchingOptions = page.getByRole('option');
  await expect(matchingOptions).toHaveCount(2);
  await expect(matchingOptions.first()).not.toContainText(/additional|reference/i);
  await page.getByRole('option', { name: /Test run.*Sep 1.*8418072e.*Commit.*Aug 15/ }).click();
  await expect(page.getByRole('combobox', { name: 'Baseline run' })).toHaveValue(/31369c4d/);
});
