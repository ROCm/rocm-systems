import { expect, test } from '@playwright/test';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
import { readChart } from './helpers/chart.js';
import { expectNoDocumentOverflow, openDashboard, ready } from './helpers/dashboard.js';

const note = 'Trend values are normalized to the latest test catalog using fixed first-success anchors for added benchmarks.';
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

test('feedback: compact trend tooltip, reachable anchors and non-selectable missing slots', async ({ page }) => {
  const errors = errorsFor(page);
  await openDashboard(page);
  const trend = page.getByTestId('performance-trend');
  const chart = trend.getByRole('img');
  for (const theme of ['light', 'dark']) {
    if (theme === 'dark') await page.getByRole('button', { name: 'Use dark theme' }).click();
    const points = await readChart(chart, (instance) => {
      const option = instance.getOption();
      const data = option.series[0].data;
      const index = data.findIndex((value, dataIndex) => value[1] != null && option.tooltip[0].formatter([{ dataIndex, value }]).includes('Normalized estimate'));
      if (index < 0) throw new Error('Fixture must include a normalized estimate');
      const gap = data.find((point) => point[1] == null && point[0] > data[0][0] && point[0] < data.at(-1)[0]);
      if (!gap) throw new Error('Fixture must include an interior missing slot');
      return { measured: instance.convertToPixel({ seriesIndex: 0 }, data[index]), gap: instance.convertToPixel({ seriesIndex: 0 }, [gap[0], data[index][1]]) };
    });
    const box = await chart.boundingBox();
    const pointerStatus = () => readChart(chart, (instance) => Object.values(instance.getModel().getComponent('axisPointer').coordSysAxesInfo.axesInfo)
      .find((info) => info.axis.dim === 'x')?.axisPointerModel.option.status);
    await page.mouse.move(box.x + points.gap[0], box.y + points.gap[1]);
    await expect.poll(pointerStatus).toBe('hide');
    await expect(trend.getByRole('status')).toHaveCount(0);
    await page.mouse.move(box.x + points.measured[0], box.y + points.measured[1]);
    await expect.poll(pointerStatus).toBe('show');
    const tooltip = await readChart(chart, (instance) => {
      const tip = [...instance.getDom().querySelectorAll('div')].find((element) => element.style.maxWidth === '240px');
      return { text: tip?.innerText, width: tip?.getBoundingClientRect().width };
    });
    expect(tooltip.text).toContain('Normalized estimate');
    expect(tooltip.text).not.toMatch(/first.success|Execution time|fictional-/i);
    expect(tooltip.width).toBeLessThanOrEqual(258);
    const selection = await trend.getByRole('status').textContent();
    await expect(trend.getByRole('group', { name: 'Inspect performance trend' }).getByRole('status')).toHaveCount(0);
    await page.mouse.move(box.x + points.gap[0], box.y + points.gap[1]);
    await expect.poll(pointerStatus).toBe('hide');
    await chart.click({ position: { x: points.gap[0], y: points.gap[1] } });
    await expect.poll(pointerStatus).toBe('hide');
    await expect.poll(() => readChart(chart, (instance) => {
      const tip = [...instance.getDom().querySelectorAll('div')].find((element) => element.style.maxWidth === '240px');
      const style = tip && getComputedStyle(tip);
      return Boolean(style && style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity) > 0);
    })).toBe(false);
    await expect(trend.getByRole('status')).toHaveText(selection);
    await trend.locator('summary').click();
    await expect(trend.getByRole('region', { name: 'Selected trend estimate anchors' })).toBeVisible();
    await expect(trend.getByRole('region', { name: 'Selected trend estimate anchors' })).toContainText('first success');
    await trend.getByRole('group', { name: 'Inspect performance trend' }).press('Escape');
    await expect(trend.getByRole('status')).toHaveCount(0);
    await expect(trend.getByRole('region', { name: 'Selected trend estimate anchors' })).toHaveCount(0);
  }
  expect(errors).toEqual([]);
});

test('feedback: cursor sweeps keep native guides visible without replacing the trend model', async ({ page }) => {
  const errors = errorsFor(page);
  await openDashboard(page);
  const chart = page.getByTestId('performance-trend').getByRole('img');
  const points = await readChart(chart, (instance) => {
    const points = instance.getOption().series[0].data.filter((point) => point[1] != null).slice(-4)
      .map((point) => instance.convertToPixel({ seriesIndex: 0 }, point.slice(0, 2)));
    instance.__feedbackOriginal = instance.setOption;
    instance.__feedbackCalls = 0;
    instance.setOption = function (...args) {
      this.__feedbackCalls++;
      return this.__feedbackOriginal.apply(this, args);
    };
    return points;
  });
  try {
    const box = await chart.boundingBox();
    const values = [];
    for (const [x, y] of points) {
      await page.mouse.move(box.x + x, box.y + y, { steps: 4 });
      await expect.poll(() => readChart(chart, (instance) => Object.values(instance.getModel().getComponent('axisPointer').coordSysAxesInfo.axesInfo)
        .find((info) => info.axis.dim === 'x')?.axisPointerModel.option.status)).toBe('show');
      values.push(await readChart(chart, (instance) => Object.values(instance.getModel().getComponent('axisPointer').coordSysAxesInfo.axesInfo)
        .find((info) => info.axis.dim === 'x').axisPointerModel.option.value));
    }
    expect(new Set(values).size).toBeGreaterThan(1);
    expect(await readChart(chart, (instance) => instance.__feedbackCalls)).toBe(0);
    expect(errors).toEqual([]);
  } finally {
    await readChart(chart, (instance) => { instance.setOption = instance.__feedbackOriginal; delete instance.__feedbackOriginal; delete instance.__feedbackCalls; });
  }
});

test('feedback: static trend explanation, interrupted solid line, blue commit links and fresh ST+MT defaults', async ({ page }) => {
  const errors = errorsFor(page);
  await openDashboard(page);
  await expect(page.getByRole('checkbox', { name: 'ST', exact: true })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: 'MT', exact: true })).toBeChecked();
  const trend = page.getByTestId('performance-trend');
  const chart = await readChart(trend.getByRole('img'), (instance) => {
    const series = instance.getOption().series[0];
    return { connectNulls: series.connectNulls, nulls: series.data.filter((point) => point[1] == null).length, left: instance.getOption().grid[0].left };
  });
  expect(chart.connectNulls).toBe(false);
  expect(chart.left).toBe(40);
  expect((await trend.getByRole('img').boundingBox()).height).toBe(320);
  expect(await trend.getByRole('group', { name: 'Inspect performance trend' }).evaluate((element) => getComputedStyle(element).marginTop)).toBe('16px');
  expect(chart.nulls).toBeGreaterThan(0);
  for (const label of ['Trailing 7 days', 'Trailing 30 days', 'Trailing 90 days', 'All available history']) {
    await trend.getByRole('button', { name: label, exact: true }).click();
    await expect(trend).toContainText(note);
    await expect(trend).toContainText('Explore original per-benchmark measurements:');
    await expect(trend.getByRole('button', { name: 'Benchmarks', exact: true })).toBeVisible();
  }
  const row = page.getByTestId('recent-runs-table').locator('tbody tr').first();
  const runId = await row.getAttribute('data-run-id');
  const publication = createSchema2Publication();
  const source = publication.runs.find((run) => run.id === runId).source;
  const commit = row.locator('td').first().getByRole('link');
  await expect(commit).toHaveAttribute('href', `${source.repository ?? 'https://github.com/ROCm/rocm-systems'}/commit/${source.commit}`);
  await expect(commit).toHaveAttribute('target', '_blank');
  await expect(commit).toHaveAttribute('rel', /noopener/);
  await expect(commit).toHaveAttribute('rel', /noreferrer/);
  await commit.focus();
  await expect(commit).toBeFocused();
  const colors = await commit.evaluate((element) => ({ link: getComputedStyle(element).color, normal: getComputedStyle(element.closest('td')).color }));
  expect(colors.link).not.toBe(colors.normal);
  await page.getByRole('checkbox', { name: 'ST', exact: true }).uncheck();
  await page.getByRole('checkbox', { name: 'MT', exact: true }).uncheck();
  await expect(trend).toContainText(note);
  await expect(trend.getByRole('button', { name: 'Benchmarks', exact: true })).toBeVisible();
  await page.reload();
  await ready(page);
  await expect(page.getByRole('checkbox', { name: 'MT', exact: true })).not.toBeChecked();
  expect(errors).toEqual([]);
});

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
  await openDashboard(page, '/?view=branch&branch=fictional/optimization-01&run=fictional-branch-01&target=gfx1250&mode=ST');
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
      const values = await rows.evaluateAll((entries, attr) => entries.map((entry) => {
        const value = entry.getAttribute(attr);
        return value === '' || value === null || value === 'unavailable' ? null : Number(value);
      }), attribute);
      expect(values.length).toBeGreaterThan(1);
      const measured = values.filter((value) => value !== null);
      expect(measured).toEqual([...measured].sort((a, b) => order === 'ascending' ? a - b : b - a));
      const firstMissing = values.indexOf(null);
      if (firstMissing >= 0) expect(values.slice(firstMissing).every((value) => value === null)).toBe(true);
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
    const different = page.getByTestId('metadata-row-sdk');
    await expect(different).toHaveAttribute('data-different', 'true');
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
