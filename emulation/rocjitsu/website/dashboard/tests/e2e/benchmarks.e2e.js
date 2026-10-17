import { fixtureRunPath } from '../fixtures/runPath.js';
import { expect, test } from '@playwright/test';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
import { formatDuration } from '../../src/utils/formatters.js';
import { clickLastCompletedChartPoint, readChart, selectedRunIds } from './helpers/chart.js';
import { ensureGridBenchmark, inspectResult, installPublication, openDashboard, ready } from './helpers/dashboard.js';

const gemm = 'Fictional GEMM';
const card = (page, name) => page.getByTestId('benchmark-grid-card').filter({ has: page.getByRole('heading', { name, exact: true }) });

test('grid picker bounds search and eight slots while draft cancel, pointer removal and empty recovery work', async ({ page }) => {
  const publication = createSchema2Publication();
  // One fictional run is enough to test a catalog larger than the visible window.
  const run = publication.runs.find((entry) => entry.id === 'fictional-develop-23');
  publication.runs = [run];
  publication.index.runFiles = [fixtureRunPath(run)];
  const catalog = publication.catalogs[run.testCatalog];
  const extras = Array.from({ length: 55 }, (_, index) => ({ id: `picker-${index}`, name: `Fictional picker ${String(index).padStart(2, '0')}`, suite: 'Triton', problem: { size: index, ...(index === 54 ? { dataType: 64 } : {}) } }));
  catalog.tests.push(...extras);
  catalog.configurations['gfx1250:ST'].push(...extras.map((entry) => entry.id));
  run.configurations.find((config) => config.target === 'gfx1250' && config.threadingMode === 'ST')
    .results.push(...extras.map((entry) => ({ testId: entry.id, status: 'failed', durationSeconds: null, error: 'Fictional picker-only diagnostic' })));
  await installPublication(page, { publication });
  await openDashboard(page, '/?view=benchmarks');
  const headings = page.getByTestId('benchmark-grid-card').getByRole('heading');
  await expect(headings).toHaveCount(4);
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
  for (const index of [54, 53, 52, 51]) {
    await picker.fill(`Fictional picker ${index}`);
    if (index === 54) {
      await picker.press('ArrowDown');
      await picker.press('Enter');
      await picker.fill('Fictional picker 54');
      await expect(scalarOption).toHaveAttribute('aria-selected', 'true');
    } else await page.getByRole('option').click();
  }
  await picker.fill('Fictional picker 50');
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
  await expect(headings).toHaveCount(4);
  const original = await headings.allTextContents();
  const retained = original.filter((name) => name !== 'Fictional decode');
  expect(retained).toHaveLength(3);
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
  expect(['ST', 'MT']).toContain(clickedPoint.mode);
  await expect(dialog.getByText('Execution mode', { exact: true }).locator('..')).toContainText(clickedPoint.mode);
  await expect(dialog.getByText('Simulator threads', { exact: true })).toHaveCount(0);
  await dialog.getByRole('button', { name: 'Close details', exact: true }).click();
  expect(await selectedRunIds(chart)).toEqual(['fictional-develop-23']);
  await page.getByRole('button', { name: 'Trailing 7 days' }).click();
  await expect(page.getByRole('button', { name: 'Trailing 7 days' })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('checkbox', { name: 'MT', exact: true }).uncheck();
  expect(await selectedRunIds(chart)).toEqual(['fictional-develop-23']);
  const series = await readChart(chart, (instance) => {
    const option = instance.getOption();
    return { zoom: option.dataZoom ?? [], lines: option.series.filter((series) => series.type === 'line').map((series) => series.name) };
  });
  expect(series).toEqual({ zoom: [], lines: ['gfx1250 ST', 'gfx950 ST'] });
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

test('keyboard result details distinguish completed zero, failed, timeout and unpublished configurations', async ({ page }) => {
  await openDashboard(page, '/?view=benchmarks');
  await page.getByRole('checkbox', { name: 'gfx950', exact: true }).check();
  await ensureGridBenchmark(page, gemm);
  await page.getByRole('button', { name: `Inspect ${gemm} results`, exact: true }).click();
  const inspector = page.getByRole('dialog');
  await inspector.getByRole('combobox', { name: `${gemm} result`, exact: true }).fill('');
  await expect(page.getByRole('option')).toHaveCount(50);
  await page.keyboard.press('Escape');
  await inspector.getByRole('button', { name: 'Close', exact: true }).click();
  for (const [name, runId, mode, status] of [
    [gemm, 'fictional-develop-14', 'ST', 'Completed'],
    ['Fictional added workload D', 'fictional-develop-08', 'ST', 'Failed'],
    ['Fictional added workload E', 'fictional-develop-08', 'MT', 'Timeout'],
  ]) {
    await ensureGridBenchmark(page, name);
    const dialog = await inspectResult(page, name, runId, 'gfx1250', mode);
    await expect(dialog.getByText(status, { exact: true })).toBeVisible();
    await expect(dialog.getByText(runId, { exact: true })).toBeVisible();
    await expect(dialog.getByText('Run Provenance', { exact: true })).toBeVisible();
    await expect(dialog.getByText('Environment', { exact: true })).toBeVisible();
    await expect(dialog.getByText('Fictional SDK', { exact: true })).toBeVisible();
    await expect(dialog.getByText('Execution mode', { exact: true }).locator('..')).toContainText(mode);
    await expect(dialog.getByText('Duration', { exact: true }).locator('..')).toContainText(status === 'Completed' ? formatDuration(0) : '—');
    if (status !== 'Completed') await expect(dialog.getByText('Fictional diagnostic', { exact: true })).toBeVisible();
    await dialog.getByRole('button', { name: 'Close details', exact: true }).click();
    await expect(dialog).toHaveCount(0);
    const parent = page.locator('[role="dialog"][aria-labelledby="inspect-results-title"]');
    await expect(parent).toBeVisible();
    await parent.getByRole('button', { name: 'Close', exact: true }).click();
    await expect(parent).toHaveCount(0);
  }
  await inspectResult(page, gemm, 'fictional-develop-12', 'gfx950', 'MT');
  // Unpublished results stay in the inspector; no nested details are invented.
  const absent = page.locator('[role="dialog"][aria-labelledby="inspect-results-title"]');
  await expect(page.getByRole('button', { name: 'Close details', exact: true })).toHaveCount(0);
  await expect(absent.getByRole('alert')).toContainText('Unavailable · gfx950 · MT · fictional-develop-12');
  await expect(absent.getByRole('alert')).toContainText('not a failed or zero-duration measurement');
  await expect(absent.getByText('Run Provenance', { exact: true })).toHaveCount(0);
});
