import raw from '../fixtures/fictional-producer-run.json' with { type: 'json' };
import { expect, test } from '@playwright/test';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { downloadSource, inspectResult, installPublication, openDashboard, ready } from './helpers/dashboard.js';

test('literal inherited-property metadata namespaces render safely in both comparison tables', async ({ page }) => {
  const publication = createMaintainedPublication();
  for (const run of publication.runs) run.provenance = {
    ...run.provenance,
    ...JSON.parse('{"__proto__":{"version":"literal-prototype"},"constructor":{"version":"literal-constructor"},"toString":{"version":"literal-method"}}'),
  };
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  await installPublication(page, { publication });
  await openDashboard(page, '/?view=compare&compareCandidate=fictional-branch-01&compareBaseline=fictional-develop-20');
  for (const namespace of ['__proto__', 'constructor', 'toString']) {
    await expect(page.getByRole('table', { name: 'Run metadata differences', exact: true }).getByText(namespace, { exact: true })).toBeVisible();
  }
  await openDashboard(page, '/?view=branch&branch=fictional/optimization-01&run=fictional-branch-01');
  const environment = page.getByTestId('branch-environment');
  await environment.locator('summary').click();
  for (const namespace of ['__proto__', 'constructor', 'toString']) await expect(environment.getByText(namespace, { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('typed definitions render in details and remain intact in filtered JSON export and reload', async ({ page }) => {
  const publication = createMaintainedPublication();
  const problem = { shape: [16, 16], options: { transpose: false, note: null }, note: null, literal: '[16,16]' };
  for (const run of publication.runs) for (const row of run.benchmark_results) if (row.id === 'a') row.problem = structuredClone(problem);
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  await installPublication(page, { publication });
  await openDashboard(page, '/?view=benchmarks&targets=gfx1250&modes=single');
  const details = await inspectResult(page, 'Fictional GEMM', 'fictional-develop-23');
  await expect(details.getByText('[16,16]', { exact: true })).toHaveCount(2);
  await expect(details.getByText('{"transpose":false,"note":null}', { exact: true })).toBeVisible();
  await expect(details.getByText('null', { exact: true })).toBeVisible();
  await details.getByRole('button', { name: 'Close details', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Close', exact: true }).click();
  await ready(page);
  expect(await downloadSource(page)).toEqual(publication);
  await page.reload();
  await ready(page);
  expect(new URL(page.url()).searchParams.getAll('modes')).toEqual(['single']);
  expect(await downloadSource(page)).toEqual(publication);
  expect(errors).toEqual([]);
});

for (const width of [1440, 390]) test(`fictional producer output keeps undated runs and renders missing provenance at ${width}px`, async ({ page }) => {
  await page.setViewportSize({ width, height: 900 });
  const run = structuredClone(raw);
  Object.assign(run.publication, { id: 'fictional-optional', source: { branch: 'develop', message: null, base: null, pull_request: { number: 12 } } });
  Object.assign(run.provenance.rocjitsu, { rocjitsu_commit_sha: null, rocjitsu_commit_timestamp: null, rocm_sdk_version: null });
  const publication = { index: { generatedAt: '2026-10-21T20:00:00Z', runFiles: ['runs/default-branch/fictional-optional.json'] }, runs: [run] };
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const { requests } = await installPublication(page, { publication });
  await openDashboard(page);
  const recent = page.getByTestId('recent-runs-table').locator('tbody tr');
  await expect(recent).toHaveCount(1);
  await expect(recent.locator('td').nth(2)).toHaveText('-');
  expect(await downloadSource(page)).toEqual(publication);
  await page.goto('/?view=compare&compareCandidate=fictional-optional&compareBaseline=fictional-optional');
  await ready(page);
  await expect(page.getByTestId('comparison-metric-comparable')).toContainText('3');
  await expect(page.getByTestId('metadata-row-machine').getByRole('cell').nth(0)).toHaveText('—');
  await expect(page.getByTestId('metadata-row-rocjitsu.workflow_commit_sha').getByRole('cell').nth(0)).toHaveText('—');
  expect(await downloadSource(page)).toEqual(publication);
  expect(requests.some((url) => url.includes('test-catalogs'))).toBe(false);
  expect(errors).toEqual([]);
});
