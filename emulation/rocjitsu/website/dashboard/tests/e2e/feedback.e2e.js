import { expect, test } from '@playwright/test';
import { expectNoDocumentOverflow, openDashboard, ready } from './helpers/dashboard.js';

const branchName = 'users/RattataKing/test-branch';
const errorsFor = (page) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  return errors;
};
async function theme(page, mode) {
  await page.evaluate((value) => localStorage.setItem('rocjitsu-color-mode', value), mode);
  await page.reload();
  await ready(page);
}

test('feedback: either empty comparison selector retains geometry on focus/open in both themes and narrow layouts', async ({ page }) => {
  const errors = errorsFor(page);
  await openDashboard(page, '/?view=compare');
  for (const mode of ['light', 'dark']) {
    await theme(page, mode);
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      for (const label of ['Baseline run', 'Candidate run']) {
        const picker = page.getByRole('combobox', { name: label, exact: true });
        await expect(picker).toHaveValue('');
        await picker.scrollIntoViewIfNeeded();
        const geometry = (element) => {
          const rect = element.getBoundingClientRect();
          const form = element.closest('.MuiFormControl-root').getBoundingClientRect();
          return { x: rect.x + scrollX, y: rect.y + scrollY, width: rect.width, height: rect.height, formHeight: form.height };
        };
        const before = await picker.evaluate(geometry);
        await picker.click();
        await expect(picker).toHaveAttribute('aria-expanded', 'true');
        await expect.poll(async () => {
          const after = await picker.evaluate(geometry);
          return Math.max(...Object.keys(before).map((key) => Math.abs(after[key] - before[key])));
        }).toBeLessThanOrEqual(1);
        await picker.press('Escape');
      }
      await expectNoDocumentOverflow(page);
    }
  }
  expect(errors).toEqual([]);
});

test('feedback: exact fictional branch, no heading count badge and leftward collapse preserve selection/search', async ({ page }) => {
  const errors = errorsFor(page);
  await openDashboard(page, '/?view=branch');
  for (const mode of ['light', 'dark']) {
    await theme(page, mode);
    const list = page.getByTestId('branch-list-panel');
    await expect(list.getByRole('heading', { name: 'Branches', exact: true })).toBeVisible();
    await expect(list.getByRole('status')).toHaveCount(0);
    const search = page.getByRole('searchbox', { name: 'Search branch, PR or SHA', exact: true });
    await search.fill('');
    await search.blur();
    await expect.poll(() => search.evaluate((input) => {
      const root = input.closest('.MuiInputBase-root').getBoundingClientRect();
      const label = input.closest('.MuiFormControl-root').querySelector('label').getBoundingClientRect();
      return Math.abs(label.y + label.height / 2 - root.y - root.height / 2);
    })).toBeLessThan(1);
    await search.click();
    await expect(search).toHaveAttribute('placeholder', 'Search branch, PR or SHA');
    await expect.poll(() => search.evaluate((input) => input.closest('.MuiFormControl-root').querySelector('label').getAttribute('data-shrink'))).toBe('true');
    await expect.poll(() => search.evaluate((input) => Number(getComputedStyle(input, '::placeholder').opacity))).toBeGreaterThan(0);
    const centeredInput = await search.evaluate((input) => {
      const root = input.closest('.MuiInputBase-root').getBoundingClientRect();
      const rect = input.getBoundingClientRect();
      return Math.abs(rect.y + rect.height / 2 - root.y - root.height / 2);
    });
    expect(centeredInput).toBeLessThan(1);
    await search.fill(branchName);
    await page.getByRole('button', { name: new RegExp(`^View branch ${branchName},`) }).click();
    const candidate = await page.getByTestId('candidate-selected-identity').textContent();
    const reference = await page.getByTestId('reference-selected-identity').textContent();
    const url = page.url();
    const detail = page.locator('#branch-detail');
    const expanded = await detail.boundingBox();
    await page.getByRole('button', { name: 'Collapse branches', exact: true }).press('Enter');
    const reopen = page.getByRole('button', { name: 'Expand branches', exact: true });
    await expect(reopen).toHaveAttribute('aria-expanded', 'false');
    await expect(list).not.toBeVisible();
    const collapsed = await detail.boundingBox();
    expect(collapsed.x).toBeLessThan(expanded.x);
    expect(collapsed.width).toBeGreaterThan(expanded.width);
    await expect(search).toHaveValue(branchName);
    await expect(page.getByTestId('candidate-selected-identity')).toHaveText(candidate);
    await expect(page.getByTestId('reference-selected-identity')).toHaveText(reference);
    expect(page.url()).toBe(url);
    await reopen.press('Enter');
    await expect(list).toBeVisible();
    await expect(page.getByRole('button', { name: 'Collapse branches', exact: true })).toHaveAttribute('aria-expanded', 'true');
    await expect(search).toHaveValue(branchName);
    await expectNoDocumentOverflow(page);
  }
  expect(errors).toEqual([]);
});

test('feedback: Abs/Pct headers sort raw deltas and full comparison navigation returns to page top', async ({ page }) => {
  const errors = errorsFor(page);
  await page.setViewportSize({ width: 1440, height: 600 });
  await openDashboard(page, '/?view=branch&branch=fictional/optimization-01&run=fictional-branch-01&target=gfx1250&mode=single');
  const region = page.getByRole('region', { name: 'Benchmark differences', exact: true });
  await expect(page.getByLabel('Sort by magnitude')).toHaveCount(0);
  const rows = region.locator('tbody tr[data-abs-change]');
  for (const [name, attribute] of [['Abs change', 'data-abs-change'], ['Pct change', 'data-pct-change']]) {
    const header = region.getByRole('columnheader', { name: new RegExp(name) });
    const sort = header.getByRole('button');
    await sort.click();
    await expect(header).toHaveAttribute('aria-sort', /ascending|descending/);
    for (let step = 0; step < 2; step++) {
      const order = await header.getAttribute('aria-sort');
      const groups = await rows.evaluateAll((entries, attr) => {
        const groups = [];
        for (const entry of entries) {
          if (entry.previousElementSibling?.querySelector('[scope="rowgroup"]')) groups.push([]);
          const value = entry.getAttribute(attr);
          groups.at(-1).push(value === '' || value === null || value === 'unavailable' ? null : Number(value));
        }
        return groups;
      }, attribute);
      expect(groups.flat().length).toBeGreaterThan(1);
      for (const values of groups) {
        const measured = values.filter((value) => value !== null);
        expect(measured).toEqual([...measured].sort((a, b) => order === 'ascending' ? a - b : b - a));
        const firstMissing = values.indexOf(null);
        if (firstMissing >= 0) expect(values.slice(firstMissing).every((value) => value === null)).toBe(true);
      }
      await sort.press('Enter');
    }
  }
  const open = page.getByRole('button', { name: 'Open full comparison ↗', exact: true });
  await open.evaluate((element) => {
    const y = element.getBoundingClientRect().top + scrollY;
    window.scrollTo({ top: Math.max(0, y - 80), behavior: 'instant' });
  });
  expect(await page.evaluate(() => scrollY)).toBeGreaterThan(0);
  await open.click();
  await expect(page.getByRole('tab', { name: 'Run Comparison', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByTestId('candidate-run-selected-identity')).toContainText('fictional-branch-01');
  await expect(page.getByTestId('baseline-run-selected-identity')).toContainText('fictional-develop-20');
  await expect.poll(() => page.evaluate(() => scrollY)).toBe(0);
  expect(errors).toEqual([]);
});

test('feedback: compared metadata highlights complete differing rows instead of Changed labels', async ({ page }) => {
  const errors = errorsFor(page);
  await openDashboard(page, '/?view=compare&compareCandidate=fictional-branch-01&compareBaseline=fictional-develop-20');
  for (const mode of ['light', 'dark']) {
    await theme(page, mode);
    const table = page.getByRole('table', { name: 'Run metadata differences', exact: true });
    await expect(table.getByText('Changed', { exact: true })).toHaveCount(0);
    const different = page.getByTestId('metadata-row-auxiliary.sdk');
    await expect(different).toHaveAttribute('data-different', 'true');
    await expect(different.getByRole('cell').nth(0)).toHaveText('test-b');
    await expect(different.getByRole('cell').nth(1)).toHaveText('test-branch');
    const backgrounds = await different.locator('th, td').evaluateAll((cells) => cells.map((cell) => getComputedStyle(cell).backgroundColor));
    expect(backgrounds).toHaveLength(3);
    expect(new Set(backgrounds).size).toBe(1);
    expect(backgrounds[0]).not.toBe('rgba(0, 0, 0, 0)');
    const same = page.getByTestId('metadata-row-catalog');
    await expect(same).toHaveAttribute('data-different', 'false');
    expect(await same.locator('th').evaluate((cell) => getComputedStyle(cell).backgroundColor)).not.toBe(backgrounds[0]);
  }
  expect(errors).toEqual([]);
});
