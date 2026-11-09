import { fixtureRunPath } from '../fixtures/runPath.js';
import { expect, test } from '@playwright/test';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { formatDuration } from '../../src/utils/formatters.js';
import { clickLastCompletedChartPoint, readChart, selectedRunIds } from './helpers/chart.js';
import { ensureGridBenchmark, inspectResult, installPublication, openDashboard, ready } from './helpers/dashboard.js';

const gemm = 'Fictional GEMM';
const card = (page, name) => page.getByTestId('benchmark-grid-card').filter({ has: page.getByRole('heading', { name, exact: true }) });

test('grid picker bounds search and eight slots while draft cancel, pointer removal and empty recovery work', async ({ page }) => {
  const publication = createMaintainedPublication();
  // One fictional run is enough to test a catalog larger than the visible window.
  const run = publication.runs.find((entry) => entry.publication.id === 'fictional-develop-23');
  publication.runs = [run];
  publication.index.runFiles = [fixtureRunPath(run)];
  const extras = Array.from({ length: 55 }, (_, index) => ({ id: `picker-${index}`, name: `Fictional picker ${String(index).padStart(2, '0')}`, suite: 'Triton', operation: 'Fictional computation', problem: { size: index, ...(index === 54 ? { dtype: 64 } : {}) } }));
  run.benchmark_results.push(...extras.map((entry) => ({ ...entry, target: 'gfx1250', thread_policy: 'single', exit_code: 1, status: 'failed', timing_results_s: [], error: 'Fictional picker-only diagnostic' })));
  await installPublication(page, { publication });
  await openDashboard(page, '/?view=benchmarks');
  const headings = page.getByTestId('benchmark-grid-card').getByRole('heading');
  await expect(headings).toHaveCount(3);
  const original = await headings.allTextContents();
  await page.getByRole('button', { name: 'Add benchmarks', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Add benchmarks', exact: true });
  const picker = dialog.getByRole('combobox', { name: 'Benchmarks to graph' });
  await picker.fill('');
  await expect(page.getByRole('option')).toHaveCount(50);
  await picker.fill('no-such-benchmark');
  await expect(page.getByRole('option')).toHaveCount(0);
  await expect(page.getByText('No benchmarks match within the selected targets / suites')).toBeVisible();
  await picker.fill('Fictional picker 54');
  const scalarOption = page.getByRole('option', { name: /^Fictional picker 54\b/ });
  await expect(scalarOption).toContainText('64');
  // Selection belongs to the option, not an unnamed nested form control.
  await expect(scalarOption.getByRole('checkbox')).toHaveCount(0);
  await expect(scalarOption.locator('input[type="checkbox"]')).toHaveCount(0);
  await scalarOption.click();
  // MUI clears search after selection; the bounded window otherwise hides #54.
  await picker.fill('Fictional picker 54');
  await expect(scalarOption).toHaveAttribute('aria-selected', 'true');
  await picker.press('Escape');
  await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
  // MUI hides the page from role locators until the exiting modal detaches.
  await expect(headings).toHaveText(original);
  await page.getByRole('button', { name: 'Add benchmarks', exact: true }).click();
  for (const index of [54, 53, 52, 51, 50]) {
    await picker.fill(`Fictional picker ${index}`);
    if (index === 54) {
      await picker.press('ArrowDown');
      await picker.press('Enter');
      await picker.fill('Fictional picker 54');
      await expect(scalarOption).toHaveAttribute('aria-selected', 'true');
    } else await page.getByRole('option').click();
  }
  await picker.fill('Fictional picker 49');
  await expect(page.getByRole('option')).toHaveAttribute('aria-disabled', 'true');
  await picker.press('ArrowDown');
  await picker.press('Enter');
  await picker.press('Escape');
  await expect(dialog.getByText(/8 of 8 benchmarks selected/)).toBeVisible();
  await dialog.getByRole('button', { name: 'Apply selection' }).click();
  await expect(page.getByTestId('benchmark-grid-card')).toHaveCount(8);
  await page.getByRole('button', { name: 'Remove Fictional picker 54 from grid', exact: true }).click();
  await expect(page.getByTestId('benchmark-grid-card')).toHaveCount(7);
  await ensureGridBenchmark(page, 'Fictional picker 54');
  await expect(page.getByTestId('benchmark-grid-card')).toHaveCount(8);
  while (await page.getByTestId('benchmark-grid-card').count()) {
    await page.getByRole('button', { name: /^Remove .* from grid$/ }).first().press('Enter');
  }
  await expect(page.getByText('No benchmark graphs selected. Add benchmarks to begin.')).toBeVisible();
  await ensureGridBenchmark(page, gemm);
  await expect(page.getByTestId('benchmark-grid-card')).toHaveCount(1);
  await expect(card(page, gemm).getByRole('img')).toBeVisible();
});

test('removed and explicitly empty workload choices survive tab remounts and data reloads', async ({ page }) => {
  await openDashboard(page, '/?view=benchmarks');
  const headings = page.getByTestId('benchmark-grid-card').getByRole('heading');
  await expect(headings).toHaveCount(3);
  const original = await headings.allTextContents();
  const retained = original.filter((name) => name !== 'Fictional decode');
  expect(retained).toHaveLength(2);
  await page.getByRole('button', { name: 'Remove Fictional decode from grid', exact: true }).click();
  await expect(headings).toHaveText(retained);

  const remountAndRefresh = async (expected) => {
    await page.getByRole('tab', { name: 'Overview', exact: true }).click();
    await page.getByRole('tab', { name: 'Benchmarks', exact: true }).click();
    await expect(headings).toHaveText(expected);
    await page.getByRole('button', { name: 'Reload all data', exact: true }).click();
    await ready(page);
    await expect(headings).toHaveText(expected);
  };
  await remountAndRefresh(retained);
  for (const name of retained) {
    await page.getByRole('button', { name: `Remove ${name} from grid`, exact: true }).press('Enter');
  }
  await expect(headings).toHaveCount(0);
  await remountAndRefresh([]);
  await expect(page.getByText('No benchmark graphs selected. Add benchmarks to begin.')).toBeVisible();
  await ensureGridBenchmark(page, gemm);
  await expect(headings).toHaveText([gemm]);
  await remountAndRefresh([gemm]);
});

test('grid point selection survives timeframe and scope changes without legacy mode or zoom controls', async ({ page }) => {
  await openDashboard(page, '/?view=benchmarks');
  await ensureGridBenchmark(page, gemm);
  const chart = card(page, gemm).getByRole('img', { name: `${gemm} duration history` });
  await expect(page.getByRole('button', { name: /^(Single|Grid|Aggregate)$/ })).toHaveCount(0);
  const clickedPoint = await clickLastCompletedChartPoint(chart);
  let dialog = page.getByRole('dialog');
  await expect(dialog.getByText('fictional-develop-23', { exact: true })).toBeVisible();
  expect(['single', 'default']).toContain(clickedPoint.mode);
  await expect(dialog.getByText('Thread policy', { exact: true }).locator('..')).toContainText(clickedPoint.mode);
  await expect(dialog.getByText('Simulator threads', { exact: true })).toHaveCount(0);
  await dialog.getByRole('button', { name: 'Close details', exact: true }).click();
  expect(await selectedRunIds(chart)).toEqual(['fictional-develop-23']);
  await page.getByRole('button', { name: 'Trailing 7 days' }).click();
  await expect(page.getByRole('button', { name: 'Trailing 7 days' })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('checkbox', { name: 'default', exact: true }).uncheck();
  expect(await selectedRunIds(chart)).toEqual(['fictional-develop-23']);
  const series = await readChart(chart, (instance) => {
    const option = instance.getOption();
    return { zoom: option.dataZoom ?? [], lines: option.series.filter((series) => series.type === 'line').map((series) => series.name) };
  });
  expect(series).toEqual({ zoom: [], lines: ['gfx1250 single', 'gfx950 single'] });
  // Re-open after an explicit clear: clicking an already-selected run toggles
  // that run off; detail inspection itself is not a promise to keep it selected.
  await page.getByRole('button', { name: 'Clear selected runs (1)', exact: true }).click();
  expect(await selectedRunIds(chart)).toEqual([]);
  await clickLastCompletedChartPoint(chart);
  dialog = page.getByRole('dialog');
  await expect(dialog.getByText('fictional-develop-23', { exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Clear selected runs (1)', exact: true }).click();
  expect(await selectedRunIds(chart)).toEqual([]);
});

test('keyboard result details distinguish positive completion, failed, timeout and unpublished configurations', async ({ page }) => {
  await openDashboard(page, '/?view=benchmarks&range=ALL');
  await page.getByRole('checkbox', { name: 'gfx950', exact: true }).check();
  await ensureGridBenchmark(page, gemm);
  await page.getByRole('button', { name: `Inspect ${gemm} results`, exact: true }).click();
  const inspector = page.getByRole('dialog');
  await inspector.getByRole('combobox', { name: `${gemm} result`, exact: true }).fill('');
  await expect(page.getByRole('option')).toHaveCount(50);
  await page.keyboard.press('Escape');
  await inspector.getByRole('button', { name: 'Close', exact: true }).click();
  for (const [name, runId, mode, status] of [
    [gemm, 'fictional-develop-14', 'single', 'Completed'],
    ['Fictional added workload D', 'fictional-develop-08', 'single', 'Failed'],
    ['Fictional added workload E', 'fictional-develop-08', 'default', 'Timeout'],
  ]) {
    await ensureGridBenchmark(page, name);
    const dialog = await inspectResult(page, name, runId, 'gfx1250', mode);
    await expect(dialog.getByText(status, { exact: true })).toBeVisible();
    await expect(dialog.getByText(runId, { exact: true })).toBeVisible();
    await expect(dialog.getByText('Run Provenance', { exact: true })).toBeVisible();
    await expect(dialog.getByText('Environment', { exact: true })).toBeVisible();
    await expect(dialog.getByText('auxiliary.sdk', { exact: true })).toBeVisible();
    await expect(dialog.getByText('Thread policy', { exact: true }).locator('..')).toContainText(mode);
    await expect(dialog.getByText('Duration', { exact: true }).locator('..')).toContainText(status === 'Completed' ? formatDuration(createMaintainedPublication().runs.find(({ publication: { id } }) => id === runId).benchmark_results.find(({ target, thread_policy, id }) => target === 'gfx1250' && thread_policy === mode && id === 'a').timing_results_s[0]) : '—');
    if (status !== 'Completed') await expect(dialog.getByText('Fictional diagnostic', { exact: true })).toBeVisible();
    await dialog.getByRole('button', { name: 'Close details', exact: true }).click();
    await expect(dialog).toHaveCount(0);
    const parent = page.locator('[role="dialog"][aria-labelledby="inspect-results-title"]');
    await expect(parent).toBeVisible();
    await parent.getByRole('button', { name: 'Close', exact: true }).click();
    await expect(parent).toHaveCount(0);
  }
  await inspectResult(page, gemm, 'fictional-develop-12', 'gfx950', 'default');
  // Unpublished results stay in the inspector; no nested details are invented.
  const absent = page.locator('[role="dialog"][aria-labelledby="inspect-results-title"]');
  await expect(page.getByRole('button', { name: 'Close details', exact: true })).toHaveCount(0);
  await expect(absent.getByRole('alert')).toContainText('Unavailable · gfx950 · default · fictional-develop-12');
  await expect(absent.getByRole('alert')).toContainText('not a failed or zero-duration measurement');
  await expect(absent.getByText('Run Provenance', { exact: true })).toHaveCount(0);
});
