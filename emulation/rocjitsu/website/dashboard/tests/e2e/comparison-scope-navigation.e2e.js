import { expect, test } from '@playwright/test';
import { openDashboard, ready } from './helpers/dashboard.js';

const mode = (page, name) => page.getByRole('checkbox', { name, exact: true });
const tab = (page, name) => page.getByRole('tab', { name, exact: true });

test('comparison filters are independent through ordinary visits, reload and history navigation', async ({ page }) => {
  await openDashboard(page, '/?targets=gfx950&modes=single&suites=Triton');
  await tab(page, 'Run Comparison').click();
  await expect(mode(page, 'single')).toBeChecked();
  await expect(mode(page, 'default')).not.toBeChecked();
  await mode(page, 'default').check();
  await mode(page, 'single').uncheck();
  expect(new URL(page.url()).searchParams.getAll('modes')).toEqual(['single']);
  expect(new URL(page.url()).searchParams.getAll('compareModes')).toEqual(['default']);
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('rocjitsu-dashboard-filters')));
  expect(stored.modes).toEqual(['single']);
  await tab(page, 'Overview').click();
  await expect(mode(page, 'single')).toBeChecked();
  await expect(mode(page, 'default')).not.toBeChecked();
  await page.goBack();
  await expect(tab(page, 'Run Comparison')).toHaveAttribute('aria-selected', 'true');
  await expect(mode(page, 'default')).toBeChecked();
  await expect(mode(page, 'single')).not.toBeChecked();
  await page.goForward();
  await expect(mode(page, 'single')).toBeChecked();
  await tab(page, 'Run Comparison').click();
  await mode(page, 'default').uncheck();
  await page.reload();
  await ready(page);
  await expect(mode(page, 'single')).not.toBeChecked();
  await expect(mode(page, 'default')).not.toBeChecked();
  await tab(page, 'Benchmarks').click();
  await expect(mode(page, 'single')).toBeChecked();
});

test('shared comparison scope takes precedence over globals while legacy links still initialize from globals', async ({ page }) => {
  await openDashboard(page, '/?view=compare&targets=gfx950&modes=single&suites=Triton&compareTargets=gfx1250&compareModes=default&compareSuites=Llama');
  await expect(mode(page, 'default')).toBeChecked();
  await expect(mode(page, 'single')).not.toBeChecked();
  await expect(mode(page, 'gfx1250')).toBeChecked();
  await expect(mode(page, 'gfx950')).not.toBeChecked();
  await tab(page, 'Overview').click();
  await expect(mode(page, 'single')).toBeChecked();
  await expect(mode(page, 'gfx950')).toBeChecked();
  await openDashboard(page, '/?view=compare&targets=gfx1250&modes=default&suites=Llama');
  await expect(mode(page, 'default')).toBeChecked();
  await expect(mode(page, 'single')).not.toBeChecked();
  await expect(mode(page, 'gfx1250')).toBeChecked();
});

test('scoped branch opening changes comparison scope and pair without changing globals', async ({ page }) => {
  await openDashboard(page, '/?view=branch&branch=fictional/optimization-01&run=fictional-branch-01&target=gfx1250&mode=default&targets=gfx950&modes=single&suites=Triton');
  await page.getByRole('button', { name: 'Open full comparison ↗', exact: true }).click();
  await expect(tab(page, 'Run Comparison')).toHaveAttribute('aria-selected', 'true');
  await expect(mode(page, 'default')).toBeChecked();
  await expect(mode(page, 'single')).not.toBeChecked();
  await expect(page.getByTestId('candidate-run-selected-identity')).toContainText('fictional-branch-01');
  expect(new URL(page.url()).searchParams.getAll('targets')).toEqual(['gfx950']);
  expect(new URL(page.url()).searchParams.getAll('compareTargets')).toEqual(['gfx1250']);
  await tab(page, 'Overview').click();
  await expect(mode(page, 'single')).toBeChecked();
  await expect(mode(page, 'default')).not.toBeChecked();
});
