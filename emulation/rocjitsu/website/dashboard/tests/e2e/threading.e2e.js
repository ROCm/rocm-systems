import { expect, test } from '@playwright/test';
import { readChart } from './helpers/chart.js';
import { serveThreadingDataset, switchThreading } from './helpers/threading.js';

test('threading scopes every tab and reconciles filters while retaining tab and search', async ({ page }) => {
  await serveThreadingDataset(page);
  await page.goto('/');
  await expect(page.getByTestId('threading-filter')).toContainText('Default');
  await expect(page.getByTestId('targets-filter')).toContainText('gfx1250');
  await page.getByRole('tab', { name: 'Run Comparison' }).click();
  const candidate = page.getByTestId('candidate-run-information');
  await expect(candidate).toContainText('Default');
  await expect(candidate).toContainText('5/5 completed');
  await switchThreading(page, 'Single-thread');
  await expect(page.getByRole('tab', { name: 'Run Comparison' })).toHaveAttribute('aria-selected', 'true');
  await expect(candidate).toContainText('Single-thread');
  await expect(candidate).toContainText('1/1 completed');
  await expect(page.getByTestId('targets-filter')).toContainText('gfx950');
  await expect(page.getByTestId('suites-filter')).toContainText('TensileLite');
  await expect(page.getByTestId('suites-filter')).not.toContainText('Triton');

  for (const tab of ['Overview', 'Benchmarks', 'Plugin Comparison', 'Failures']) {
    await page.getByRole('tab', { name: tab, exact: true }).click();
    await switchThreading(page, 'Default');
    await expect(page.getByRole('tab', { name: tab, exact: true })).toHaveAttribute('aria-selected', 'true');
    await switchThreading(page, 'Single-thread');
    await expect(page.getByRole('tab', { name: tab, exact: true })).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByTestId('targets-filter')).toContainText('gfx950');
    await expect(page.getByRole('dialog')).toHaveCount(0);
  }
  await page.getByRole('tab', { name: 'Overview', exact: true }).click();
  const search = page.getByRole('textbox', { name: /search/i });
  await search.fill('keep this search');
  await switchThreading(page, 'Default');
  await expect(search).toHaveValue('keep this search');
});

test('threading preserves intentionally empty filter selections', async ({ page }) => {
  await serveThreadingDataset(page);
  await page.goto('/');
  await page.getByLabel('Targets').click();
  await page.getByRole('option', { name: /Check all targets/ }).click();
  await page.keyboard.press('Escape');
  await switchThreading(page, 'Single-thread');
  await expect(page.getByTestId('targets-filter').locator('[data-responsive-tag]')).toHaveCount(0);
});

test('single-only datasets select Single-thread and omit unavailable Default', async ({ page }) => {
  await serveThreadingDataset(page, ['single']);
  await page.goto('/');
  const mode = page.getByRole('combobox', { name: 'Threading', exact: true });
  await expect(page.getByTestId('threading-filter')).toContainText('Single-thread');
  await mode.click();
  await expect(page.getByRole('option', { name: 'Default', exact: true })).toHaveCount(0);
});

test('legacy-only data is a neutral empty history without catalog requests', async ({ page }) => {
  const catalogRequests = [];
  page.on('request', (request) => {
    if (request.url().includes('/test-catalogs/')) catalogRequests.push(request.url());
  });
  await page.route('**/data/index.json', (route) => route.fulfill({
    json: { generatedAt: '2026-10-06T00:00:00Z', runFiles: ['runs/legacy.json'] },
  }));
  await page.route('**/data/runs/legacy.json', (route) => route.fulfill({
    json: { id: 'legacy', testCatalog: 'test-catalogs/missing.json' },
  }));
  await page.goto('/');
  await expect(page.getByText('No supported benchmark runs yet', { exact: true })).toBeVisible();
  await expect(page.getByTestId('dashboard-data-error')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeEnabled();
  expect(catalogRequests).toEqual([]);
});

test('switching modes clears pending comparisons and opens details from the new history', async ({ page }) => {
  await serveThreadingDataset(page);
  await page.goto('/');
  const compare = page.getByRole('button', { name: /^Compare [a-f0-9]+$/ }).first();
  await compare.click();
  await expect(page.getByText('Candidate selected', { exact: true })).toBeVisible();
  const openResult = () => page.getByTestId('latest-results')
    .getByRole('button', { name: /^Open .* result details/ }).first().click();
  await openResult();
  await expect(page.getByRole('dialog').getByText('Default', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Close details', exact: true }).click();
  await switchThreading(page, 'Single-thread');
  await expect(page.getByText('Candidate selected', { exact: true })).toHaveCount(0);
  await openResult();
  await expect(page.getByRole('dialog').getByText('Single-thread', { exact: true })).toBeVisible();
  await expect(page.getByRole('dialog').getByText('Default', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Close details', exact: true }).click();
  await switchThreading(page, 'Default');
  await expect(compare).toHaveAttribute('aria-pressed', 'false');
});

test('both threading modes show independent sections on every tab and can be cleared', async ({ page }) => {
  await serveThreadingDataset(page);
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Threading', exact: true }).click();
  await page.getByRole('option', { name: 'Single-thread', exact: true }).click();
  await page.keyboard.press('Escape');
  await page.getByRole('combobox', { name: 'Targets', exact: true }).click();
  await page.getByRole('option', { name: /Check all targets/ }).click();
  await page.keyboard.press('Escape');
  const defaults = page.getByRole('region', { name: 'Default threading results', exact: true });
  const singles = page.getByRole('region', { name: 'Single-thread threading results', exact: true });
  for (const tab of ['Overview', 'Benchmarks', 'Run Comparison', 'Plugin Comparison', 'Failures']) {
    await page.getByRole('tab', { name: tab, exact: true }).click();
    await expect(defaults).toBeVisible();
    await expect(singles).toBeVisible();
    if (tab === 'Run Comparison') {
      await expect(defaults.getByTestId('candidate-run-information')).toContainText('5/5 completed');
      await expect(singles.getByTestId('candidate-run-information')).toContainText('1/1 completed');
    }
    const duplicateIds = await page.evaluate(() => {
      const ids = [...document.querySelectorAll('[id]')].map((node) => node.id);
      return ids.filter((id, index) => ids.indexOf(id) !== index);
    });
    expect(duplicateIds).toEqual([]);
  }
  await page.getByRole('combobox', { name: 'Threading', exact: true }).click();
  await page.getByRole('option', { name: /Check all threading/ }).click();
  await page.keyboard.press('Escape');
  await expect(page.getByText('Select a threading mode to view benchmark results.')).toBeVisible();
  await expect(defaults).toHaveCount(0);
  await expect(singles).toHaveCount(0);
  await switchThreading(page, 'Single-thread');
  await expect(singles).toBeVisible();
  await expect(page.getByTestId('targets-filter')).toContainText('gfx950');
  await expect(page.getByTestId('suites-filter')).toContainText('TensileLite');
});


test('main trend overlays both threading modes on one graph', async ({ page }) => {
  await serveThreadingDataset(page);
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Threading', exact: true }).click();
  await page.getByRole('option', { name: 'Single-thread', exact: true }).click();
  await page.keyboard.press('Escape');
  await page.getByRole('combobox', { name: 'Targets', exact: true }).click();
  await page.getByRole('option', { name: /Check all targets/ }).click();
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('performance-trend')).toHaveCount(1);
  const graph = page.getByTestId('performance-trend').getByRole('img');
  expect((await graph.boundingBox()).height).toBeGreaterThanOrEqual(278);
  for (const range of ['All available history', 'Trailing 7 days', 'Trailing 24 hours']) {
    await page.getByRole('button', { name: range, exact: true }).click();
    const series = await readChart(graph, (chart) => chart.getOption().series
      .filter((item) => item.type === 'line' && item.data.some((point) => Number.isFinite(point[1])))
      .map((item) => ({ name: item.name, style: item.lineStyle.type, values: item.data.map((point) => point[1]) })));
    expect(series.map((item) => item.name)).toEqual(['gfx1250 · Default', 'gfx950 · Single-thread']);
    expect(series.map((item) => item.style)).toEqual(['solid', 'dashed']);
    expect(series[0].values.some(Number.isFinite)).toBe(true);
    expect(series[1].values.some(Number.isFinite)).toBe(true);
    const tooltip = await readChart(graph, (chart) => {
      const option = chart.getOption();
      const points = option.series.flatMap((item) => {
        const index = item.data.findIndex((point) => Number.isFinite(point[1]));
        return index < 0 ? [] : [{ seriesName: item.name, value: item.data[index], dataIndex: index, marker: '' }];
      });
      return option.tooltip[0].formatter(points);
    });
    expect(tooltip).toContain('Test catalog · threading-default');
    expect(tooltip).toContain('Test catalog · threading-single');
  }
  await switchThreading(page, 'Default');
  await expect(page.getByTestId('performance-trend')).toHaveCount(1);
  const names = await readChart(page.getByTestId('performance-trend').getByRole('img'),
    (chart) => chart.getOption().series.map((item) => item.name));
  expect(names.every((name) => !name.includes('Single-thread'))).toBe(true);
});


test('combined normalization link opens complete aggregate histories for every mode', async ({ page }) => {
  const resources = await serveThreadingDataset(page);
  const oldRun = resources.get('runs/threading-default-0.json');
  const oldCatalog = structuredClone(resources.get(oldRun.testCatalog));
  oldCatalog.id = 'threading-default-old';
  oldCatalog.tests = oldCatalog.tests.slice(1);
  const ids = new Set(oldCatalog.tests.map((item) => item.id));
  oldCatalog.targets.gfx1250 = oldCatalog.targets.gfx1250.filter((id) => ids.has(id));
  oldRun.targets[0].results = oldRun.targets[0].results.filter((result) => ids.has(result.testId));
  oldRun.testCatalog = 'test-catalogs/threading-default-old.json';
  resources.set(oldRun.testCatalog, oldCatalog);
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Threading', exact: true }).click();
  await page.getByRole('option', { name: 'Single-thread', exact: true }).click();
  await page.keyboard.press('Escape');
  await page.getByRole('combobox', { name: 'Targets', exact: true }).click();
  await page.getByRole('option', { name: /Check all targets/ }).click();
  await page.keyboard.press('Escape');
  const sections = ['Default', 'Single-thread'].map((mode) => page.getByRole('region', { name: `${mode} threading results`, exact: true }));
  const openNormalized = () => page.getByTestId('performance-trend-normalization-note').getByRole('button', { name: 'Benchmarks', exact: true }).click();
  await openNormalized();
  for (const section of sections) {
    await expect(section.getByRole('button', { name: 'Aggregate', exact: true })).toHaveAttribute('aria-pressed', 'true');
  }
  for (const section of sections) {
    await page.getByRole('tab', { name: 'Overview', exact: true }).click();
    await section.getByRole('button', { name: /^View run .* in Benchmark Explorer$/ }).first().click();
    await expect(section.getByRole('button', { name: 'Clear selected runs (1)' })).toBeVisible();
  }
  await page.getByRole('tab', { name: 'Overview', exact: true }).click();
  await openNormalized();
  for (const section of sections) {
    await expect(section.getByRole('button', { name: 'Aggregate', exact: true })).toHaveAttribute('aria-pressed', 'true');
    await expect(section.getByRole('button', { name: 'Clear selected runs (0)' })).toBeDisabled();
  }
});


test('combined history hides totals for modes with no runs in the selected range', async ({ page }) => {
  const resources = await serveThreadingDataset(page);
  const index = resources.get('index.json');
  index.runFiles = index.runFiles.filter((path) => path !== 'runs/threading-single-1.json');
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Threading', exact: true }).click();
  await page.getByRole('option', { name: 'Single-thread', exact: true }).click();
  await page.keyboard.press('Escape');
  await page.getByRole('combobox', { name: 'Targets', exact: true }).click();
  await page.getByRole('option', { name: /Check all targets/ }).click();
  await page.keyboard.press('Escape');
  const single = page.getByTestId('history-summary-single');
  const defaults = page.getByTestId('history-summary-default');
  await page.getByRole('button', { name: 'Trailing 24 hours', exact: true }).click();
  await expect(single).toContainText('No runs in this range');
  await expect(single).not.toContainText('8m 20.0s');
  await expect(defaults).toContainText('50.0s');
  await expect(defaults).not.toContainText('No runs in this range');
  await page.getByRole('button', { name: 'All available history', exact: true }).click();
  await expect(single).toContainText('8m 20.0s');
  await expect(single).not.toContainText('No runs in this range');
});
