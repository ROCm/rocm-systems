import { expect, test } from '@playwright/test';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
import { benchmarkData } from '../fixtures/publishedData.js';
import { selectPublishedBranches } from '../../src/data/branchSelectors.js';
import { expectNoDocumentOverflow, openDashboard, selectRun } from './helpers/dashboard.js';

const branchButton = (page, number) => page.getByRole('button', { name: new RegExp(`^View branch fictional/optimization-${number},`) });
const results = (page) => page.getByRole('region', { name: 'Benchmark differences', exact: true });

test('published branch search and PR filters preserve exact selection through empty matches', async ({ page }) => {
  await openDashboard(page, '/?view=branch');
  const list = page.getByTestId('branch-list-panel');
  const dataset = benchmarkData;
  const branches = selectPublishedBranches(dataset);
  const branchRows = list.getByRole('button', { name: /^View branch / });
  await expect(list.getByRole('status')).toHaveCount(0);
  await expect(branchRows).toHaveCount(branches.length);
  await expect(branchRows.first()).toContainText(branches[0].branch);
  const search = page.getByRole('searchbox', { name: 'Search branch, PR or SHA' });
  for (const query of ['optimization-01', '#1000', createSchema2Publication().runs.find((run) => run.id === 'fictional-branch-01').source.commit]) {
    await search.fill(query);
    await expect(branchRows).toHaveCount(1);
    await expect(branchButton(page, '01')).toBeVisible();
  }
  await branchButton(page, '01').press('Enter');
  await expect(page.getByTestId('candidate-selected-identity')).toContainText('fictional-branch-01');
  await expect(page.getByTestId('branch-reference-reason')).toContainText('Exact develop base');
  await expect(page.getByTestId('reference-selected-identity')).toContainText('fictional-develop-20');
  await search.fill('no-such-published-branch');
  await expect(list).toContainText('No matching branches');
  await expect(page.getByTestId('candidate-selected-identity')).toContainText('fictional-branch-01');
  await list.getByRole('button', { name: 'Clear filters' }).click();
  await page.getByLabel('Pull request', { exact: true }).selectOption('with-pr');
  await expect(branchRows).toHaveCount(selectPublishedBranches(dataset, { pr: 'with-pr' }).length);
  await page.getByLabel('Pull request', { exact: true }).selectOption('no-pr');
  await expect(branchRows).toHaveCount(selectPublishedBranches(dataset, { pr: 'no-pr' }).length);
  await expect(branchButton(page, '01')).toHaveCount(0);
  await page.getByRole('button', { name: 'Clear filters', exact: true }).click();
  await expect(branchRows).toHaveCount(branches.length);
});

test('branch matrix, diagnostics, manual reference and full comparison share one exact pair and scope', async ({ page }) => {
  await openDashboard(page, '/?view=branch&targets=&suites=&modes=');
  await branchButton(page, '02').click();
  await expect(page.getByTestId('branch-reference-reason')).toContainText('Latest develop execution completed before this candidate');
  await expect(page.getByTestId('reference-selected-identity')).toContainText('fictional-develop-21');
  const missing = page.getByTestId('branch-config-gfx1250-MT');
  await expect(missing).toHaveAttribute('data-comparison-state', 'unavailable');
  await missing.press('Enter');
  await expect(missing).toHaveAttribute('aria-pressed', 'true');
  await expect(results(page)).toBeFocused();
  await expect(results(page)).toContainText('Candidate configuration not published');
  await page.getByTestId('branch-exclusions').locator('summary').click();
  await expect(page.getByTestId('branch-exclusions')).toContainText('Configuration not published');

  await branchButton(page, '03').click();
  await page.getByTestId('branch-config-gfx1250-ST').click();
  await expect(results(page).getByRole('status')).toContainText('3 matched · 1 excluded');
  await expect(page.getByTestId('branch-exclusions')).toContainText('Candidate diagnostic: Fictional branch failure');
  await branchButton(page, '01').click();
  await selectRun(page, 'Reference', 'fictional-develop-00');
  await expect(page.getByTestId('branch-reference-reason')).toContainText('Manual reference');
  await expect(page.getByTestId('branch-exclusions')).toContainText('Not in this catalog');
  expect(new URL(page.url()).searchParams.get('manual')).toBe('1');
  await page.getByRole('button', { name: 'Restore automatic base' }).click();
  await expect(page.getByTestId('reference-selected-identity')).toContainText('fictional-develop-20');
  await page.getByTestId('branch-config-gfx950-MT').click();
  await expect(page.getByTestId('branch-selected-configuration')).toHaveText('gfx950MT');
  await results(page).getByRole('combobox', { name: 'Suites', exact: true }).click();
  await page.getByRole('option', { name: 'Llama', exact: true }).click();
  await page.keyboard.press('Escape');
  await expect(results(page).getByRole('status')).toContainText('2 matched · 0 excluded');
  await expect(page.getByLabel('Sort by magnitude')).toHaveCount(0);
  const percent = results(page).getByRole('columnheader', { name: /Pct change/ });
  await percent.getByRole('button').click();
  await expect(percent).toHaveAttribute('aria-sort', /ascending|descending/);
  await page.getByRole('searchbox', { name: 'Search benchmarks' }).fill('no-such-workload');
  await expect(results(page).getByRole('status')).toContainText('0 matched · 0 excluded');
  await expect(results(page)).toContainText('No benchmarks in the selected suites or search');
  await page.getByRole('searchbox', { name: 'Search benchmarks' }).fill('');
  await page.getByRole('button', { name: 'Open full comparison ↗' }).click();
  await expect(page.getByRole('tab', { name: 'Run Comparison' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByTestId('candidate-run-selected-identity')).toContainText('fictional-branch-01');
  await expect(page.getByTestId('baseline-run-selected-identity')).toContainText('fictional-develop-20');
  const params = new URL(page.url()).searchParams;
  expect(params.getAll('targets')).toEqual(['gfx950']);
  expect(params.getAll('modes')).toEqual(['MT']);
  expect(params.getAll('suites')).toEqual(['Triton']);
  await expect(page.getByTestId('comparison-metric-comparable').getByText('2', { exact: true })).toBeVisible();
  await expect(page.getByTestId('metadata-row-sdk')).toHaveAttribute('data-different', 'true');
  await expect(page.getByTestId('metadata-row-sdk')).not.toContainText('Changed');
  await page.goBack();
  await expect(page.getByRole('tab', { name: 'Branch Runs' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByTestId('candidate-selected-identity')).toContainText('fictional-branch-01');
  await expect(page.getByTestId('branch-config-gfx950-MT')).toHaveAttribute('aria-pressed', 'true');
  await page.goForward();
  await expect(page.getByTestId('candidate-run-selected-identity')).toContainText('fictional-branch-01');
});

test('branch layout stays within the document at normal 1280px and CSS zoom 2 at 1440px (not native browser zoom)', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await openDashboard(page, '/?view=branch');
  await expect(page.getByTestId('branch-list-panel')).toBeVisible();
  await expect(page.locator('#branch-detail')).toBeVisible();
  await expectNoDocumentOverflow(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.locator('body').evaluate((body) => { body.style.zoom = '2'; });
  await expect(page.locator('body')).toHaveCSS('zoom', '2');
  await expectNoDocumentOverflow(page);
  await expect(page.getByTestId('branch-config-gfx1250-ST')).toBeVisible();
  await page.getByTestId('branch-config-gfx1250-ST').press('Enter');
  await expect(page.getByTestId('branch-config-gfx1250-ST')).toHaveAttribute('aria-pressed', 'true');
});

test('missing published identities stay inspectable instead of being replaced by defaults', async ({ page }) => {
  await openDashboard(page, '/?view=branch&branch=fictional/optimization-01&run=fictional-branch-01&reference=fictional-missing&manual=1&target=gfx950&mode=MT&detail=1');
  await expect(page.getByTestId('reference-selected-identity')).toContainText('fictional-missing');
  await expect(page.getByTestId('branch-reference-reason')).toContainText('Saved reference attempt is unavailable');
  await expect(page.getByTestId('branch-open-comparison')).toBeDisabled();
  await page.reload();
  await expect(page.getByTestId('reference-selected-identity')).toContainText('fictional-missing');
  expect(new URL(page.url()).searchParams.get('reference')).toBe('fictional-missing');
  await page.goto('/?view=compare&compareCandidate=fictional-branch-01&compareBaseline=fictional-missing');
  await expect(page.getByText(/Selected baseline is unavailable · fictional-missing/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Swap', exact: true })).toBeDisabled();
  await expect(page.getByTestId('candidate-run-selected-identity')).toContainText('fictional-branch-01');
});
