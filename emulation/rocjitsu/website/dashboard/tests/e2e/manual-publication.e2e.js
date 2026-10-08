import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';

const readRun = (name) => JSON.parse(readFileSync(
  new URL(`../fixtures/data/runs/${name}.json`, import.meta.url), 'utf8',
));
const official = readRun('benchmark-202608311945-31369c4d');
const previous = readRun('benchmark-202608311310-9f774d29');
const manual = readRun('benchmark-202609010115-8418072e-manual');

function manualRun(id, branch, commit, completedAt) {
  const run = structuredClone(official);
  run.id = id;
  run.comparisonId = id;
  run.source = { branch, commit, committedAt: completedAt, message: 'Manual topic validation' };
  run.execution = { ...run.execution, completedAt, trigger: 'manual' };
  run.targets.forEach((target) => {
    target.results[0] = {
      ...target.results[0], durationSeconds: null, status: 'failed', error: 'Manual-only failure',
    };
  });
  return run;
}

const topic = manualRun('manual-topic', 'feature/manual-benchmarks', 'a'.repeat(40), '2026-09-02T01:15:00.000Z');
const sameCommit = manualRun('manual-same-commit', 'develop', official.source.commit, '2026-09-01T02:15:00.000Z');
sameCommit.source = structuredClone(official.source);

async function serveRuns(page, runs) {
  const files = new Map(runs.map((run) => [`runs/${run.id}.json`, run]));
  await page.route('**/data/index.json*', async (route) => {
    const response = await route.fetch();
    const index = await response.json();
    await route.fulfill({ response, json: { ...index, runFiles: [...files.keys()] } });
  });
  await page.route('**/data/runs/*.json*', async (route) => {
    const path = new URL(route.request().url()).pathname.split('/data/')[1];
    await route.fulfill({ status: 200, contentType: 'application/json', json: files.get(path) });
  });
}

async function chooseRun(page, label, query, optionName) {
  const input = page.getByRole('combobox', { name: label });
  await input.fill(query);
  await page.getByRole('option', { name: optionName }).click();
}

test('manual branch and same-commit results appear only in Run Comparison', async ({ page }) => {
  const plugins = ['asan', 'tsan', 'ubsan'].map((plugin) => readRun(`benchmark-202608311945-31369c4d-${plugin}`));
  // Include a complete manual plugin group newer than the official experiment as well.
  const manualPlugins = plugins.map((plugin) => ({
    ...structuredClone(plugin), id: `manual-topic-${plugin.plugin.id}`, comparisonId: topic.comparisonId,
    source: topic.source, execution: topic.execution,
  }));
  await serveRuns(page, [previous, official, ...plugins, manual, sameCommit, topic, ...manualPlugins]);
  await page.goto('/');

  await expect(page.getByTestId('dashboard-data-error')).toHaveCount(0);
  await expect(page.getByTestId('latest-commit-run')).toContainText('31369c4d');
  const recent = page.getByTestId('recent-runs-table');
  await expect(recent.locator('tbody tr')).toHaveCount(2);
  await expect(recent).not.toContainText('Manual');
  await expect(recent).not.toContainText('aaaaaaaa');
  await expect(page.getByTestId('latest-results')).not.toContainText('Manual-only failure');

  await page.getByRole('tab', { name: 'Benchmarks' }).click();
  await expect(page.getByTestId('historical-records')).not.toContainText('Manual topic validation');
  await page.getByRole('tab', { name: /Failures/ }).click();
  await expect(page.getByTestId('failure-case')).toHaveCount(0);
  await expect(page.getByText('Manual-only failure', { exact: true })).toHaveCount(0);
  await page.getByRole('tab', { name: 'Plugin Comparison' }).click();
  const experiment = page.getByRole('combobox', { name: 'Comparison experiment' });
  await expect(experiment).toHaveText(/31369c4d/);
  await experiment.click();
  await expect(page.getByRole('option')).toHaveCount(1);
  await expect(page.getByRole('option').filter({ hasText: 'aaaaaaaa' })).toHaveCount(0);
  await page.keyboard.press('Escape');

  await page.getByRole('tab', { name: 'Run Comparison' }).click();
  await expect(page.getByRole('combobox', { name: 'Candidate run' })).toHaveValue(/aaaaaaaa.*feature\/manual-benchmarks.*Manual/);
  await expect(page.getByRole('combobox', { name: 'Baseline run' })).toHaveValue(/31369c4d.*develop.*Auto/);
  await expect(page.getByTestId('candidate-run-information')).toContainText('feature/manual-benchmarks');
  await expect(page.getByTestId('candidate-run-information')).toContainText('Manual');
  await expect(page.getByRole('row', { name: /GEMM FP16 1024³.*Failed.*Completed/ })).toBeVisible();

  await chooseRun(page, 'Baseline run', 'Manual', /31369c4d.*Manual/);
  await expect(page.getByTestId('baseline-run-information')).toContainText('Manual');
  await page.getByRole('button', { name: 'Swap', exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Candidate run' })).toHaveValue(/31369c4d.*Manual/);
  await expect(page.getByRole('combobox', { name: 'Baseline run' })).toHaveValue(/aaaaaaaa/);

  await chooseRun(page, 'Candidate run', '8418072e', /8418072e.*Manual/);
  // Changing the candidate preserves the explicitly selected manual baseline.
  await expect(page.getByRole('combobox', { name: 'Baseline run' })).toHaveValue(/aaaaaaaa/);
});

test('manual-only publication loads with empty official views and usable comparison filters', async ({ page }) => {
  await serveRuns(page, [manual, topic]);
  await page.goto('/');

  await expect(page.getByTestId('dashboard-data-error')).toHaveCount(0);
  await expect(page.getByTestId('latest-commit-run').getByText('—', { exact: true })).toHaveCount(2);
  await expect(page.getByTestId('recent-runs-table')).not.toContainText('aaaaaaaa');
  await expect(page.getByTestId('latest-results')).not.toContainText('Manual-only failure');
  await page.getByRole('tab', { name: 'Benchmarks' }).click();
  await expect(page.getByTestId('historical-records')).not.toContainText('Manual topic validation');
  await page.getByRole('tab', { name: /Failures/ }).click();
  await expect(page.getByTestId('failure-case')).toHaveCount(0);
  await page.getByRole('tab', { name: 'Plugin Comparison' }).click();
  await expect(page.getByTestId('plugin-comparison-empty')).toBeVisible();

  await page.getByRole('tab', { name: 'Run Comparison' }).click();
  await expect(page.getByRole('combobox', { name: 'Candidate run' })).toHaveValue(/aaaaaaaa/);
  await expect(page.getByRole('combobox', { name: 'Baseline run' })).toHaveValue('');
  await expect(page.getByLabel('Targets')).toBeEnabled();
  await expect(page.getByLabel('Suites')).toBeEnabled();
  await page.getByLabel('Targets').click();
  await expect(page.getByRole('option', { name: 'gfx950' })).toBeVisible();
  await page.keyboard.press('Escape');
  await page.getByLabel('Suites').click();
  await expect(page.getByRole('option', { name: 'Triton' })).toBeVisible();
  await page.keyboard.press('Escape');
  await chooseRun(page, 'Baseline run', '8418072e', /8418072e.*Manual/);
  await expect(page.getByRole('img', { name: 'Performance change by benchmark comparison chart' })).toBeVisible();
});

test('Run Comparison restores manual-exclusive filters after visiting official views', async ({ page }) => {
  const catalog = JSON.parse(readFileSync(
    new URL('../fixtures/data/test-catalogs/rocjitsu-core-v2.json', import.meta.url), 'utf8',
  ));
  catalog.id = 'manual-exclusive-catalog';
  catalog.tests.push({ id: 'manual-exclusive-test', suite: 'ManualSuite', name: 'Manual benchmark', problem: {} });
  catalog.targets.gfx9999 = ['manual-exclusive-test'];
  const exclusive = structuredClone(topic);
  exclusive.testCatalog = 'test-catalogs/manual-exclusive-catalog.json';
  exclusive.targets.push({
    id: 'gfx9999',
    results: [{ testId: 'manual-exclusive-test', status: 'completed', durationSeconds: 1, error: null }],
  });
  await page.route('**/data/test-catalogs/manual-exclusive-catalog.json*', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', json: catalog });
  });
  await serveRuns(page, [previous, official, exclusive]);
  await page.goto('/');
  await expect(page.getByTestId('latest-commit-run')).toContainText('31369c4d');
  await page.getByRole('tab', { name: 'Run Comparison' }).click();

  for (const [label, option] of [['Targets', 'gfx9999'], ['Suites', 'ManualSuite']]) {
    const filter = page.getByTestId(`${label.toLowerCase()}-filter`);
    await filter.hover();
    await page.getByLabel(label).focus();
    await filter.getByRole('button', { name: 'Clear', exact: true }).click();
    await page.getByLabel(label).click();
    await page.getByRole('option', { name: option, exact: true }).click();
    await page.keyboard.press('Escape');
  }
  await expect(page.getByTestId('targets-filter').locator('[data-responsive-tag]')).toHaveText('gfx9999');
  await expect(page.getByTestId('suites-filter').locator('[data-responsive-tag]')).toHaveText('ManualSuite');

  await page.getByRole('tab', { name: 'Overview' }).click();
  await expect(page.getByTestId('targets-filter').locator('[data-responsive-tag]')).toHaveText('gfx1250');
  await expect(page.getByTestId('suites-filter').locator('[data-responsive-tag]')).toHaveText(['DeepSeek', 'TensileLite', 'Triton']);
  await expect(page.getByTestId('latest-results').getByText('GEMM FP16 1024³', { exact: true })).toBeVisible();
  await expect(page.getByTestId('metric-card-total-duration')).not.toContainText('—');

  for (const [label, excluded, count] of [['Targets', 'gfx9999', 2], ['Suites', 'ManualSuite', 3]]) {
    await page.getByLabel(label).click();
    await expect(page.getByRole('option', { name: excluded, exact: true })).toHaveCount(0);
    const checkAll = page.getByRole('option', { name: new RegExp(`Check all ${label.toLowerCase()}`) });
    // Suites are already all selected; toggle off and back on to verify the supported set.
    if (label === 'Suites') await checkAll.click();
    await checkAll.click();
    await expect(checkAll).toContainText(`${count} of ${count} selected`);
    await expect(checkAll.getByRole('checkbox')).toBeChecked();
    await page.keyboard.press('Escape');
  }

  await page.getByRole('tab', { name: 'Run Comparison' }).click();
  for (const [label, option] of [['Targets', 'gfx9999'], ['Suites', 'ManualSuite']]) {
    await page.getByLabel(label).click();
    await expect(page.getByRole('option', { name: option, exact: true })).toBeVisible();
    await page.keyboard.press('Escape');
  }
  await expect(page.getByTestId('targets-filter').locator('[data-responsive-tag]')).toHaveText('gfx9999');
  await expect(page.getByTestId('suites-filter').locator('[data-responsive-tag]')).toHaveText('ManualSuite');
  await expect(page.getByRole('row', { name: /Manual benchmark.*Completed/ })).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Candidate run' })).toHaveValue(/aaaaaaaa.*Manual/);
});
