import { expect, test } from '@playwright/test';
import { openDashboard, ready } from './helpers/dashboard.js';

const options = ['gfx1250', 'gfx950', 'ST', 'MT', 'Triton', 'Llama'];
async function expectScope(page, selected) {
  for (const name of options) {
    await expect(page.getByRole('checkbox', { name, exact: true })).toBeChecked({ checked: selected.includes(name) });
  }
}

test('fresh scope selects every published target, suite and mode without persisting bootstrap emptiness', async ({ page }) => {
  await openDashboard(page);
  await expectScope(page, options);
  await page.reload();
  await ready(page);
  await expectScope(page, options);
});

test('stored scope stays narrow and explicit empty URL scope takes precedence after reload', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('rocjitsu-dashboard-filters', JSON.stringify({ targets: ['gfx950'], suites: ['Triton'], modes: ['MT'] })));
  await openDashboard(page);
  await expectScope(page, ['gfx950', 'Triton', 'MT']);
  await openDashboard(page, '/?targets=&suites=&modes=');
  await expectScope(page, []);
  await page.reload();
  await ready(page);
  await expectScope(page, []);
});
