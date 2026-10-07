import { expect, test } from '@playwright/test';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
import { formatDuration } from '../../src/utils/formatters.js';
import { readChart } from './helpers/chart.js';
import { installPublication, openDashboard, ready, selectRun } from './helpers/dashboard.js';

const chartFor = (page) => page.getByRole('img', { name: 'Performance change by benchmark comparison chart' });
const bars = (page) => readChart(chartFor(page), (instance) => instance.getOption().series[0].data.map((point) => ({
  id: point.comparison.candidateTest.testId, delta: point.value,
  candidate: point.comparison.candidateTest.durationSeconds,
  baseline: point.comparison.baselineTest.durationSeconds,
})));
const metricValue = (page, name) => page.getByTestId(`comparison-metric-${name}`).locator(':scope > .MuiTypography-root').nth(1);

async function expectPair(page, candidate, baseline) {
  await expect(page.getByTestId('candidate-run-selected-identity')).toContainText(candidate);
  await expect(page.getByTestId('baseline-run-selected-identity')).toContainText(baseline);
  await expect.poll(() => {
    const params = new URL(page.url()).searchParams;
    return [params.get('compareCandidate'), params.get('compareBaseline')];
  }).toEqual([candidate, baseline]);
}

test('comparison swap is atomic and browser history restores the pair, totals and exclusions', async ({ page }) => {
  await page.addInitScript(() => {
    const push = window.history.pushState.bind(window.history);
    window.__pairHistory = [];
    window.history.pushState = (state, unused, url) => {
      window.__pairHistory.push(String(url));
      return push(state, unused, url);
    };
  });
  await openDashboard(page, '/?view=compare&compareCandidate=fictional-develop-08&compareBaseline=fictional-develop-00&targets=gfx1250&modes=ST&campaign=kept#pair');
  await expectPair(page, 'fictional-develop-08', 'fictional-develop-00');
  const original = await bars(page);
  expect(original).toHaveLength(2);
  await expect(metricValue(page, 'comparable')).toHaveText('2');
  await expect(metricValue(page, 'not-comparable')).toHaveText('3');
  const exclusions = page.getByRole('region', { name: 'Excluded results scroll area' });
  await expect(exclusions.getByRole('row', { name: /Fictional added workload D/ })).toContainText('Failed');
  await expect(exclusions.getByRole('row', { name: /Fictional added workload E/ })).toContainText('Timeout');
  const candidateTotal = formatDuration(original.reduce((sum, item) => sum + item.candidate, 0));
  const baselineTotal = formatDuration(original.reduce((sum, item) => sum + item.baseline, 0));
  await expect(metricValue(page, 'candidate-total')).toHaveText(candidateTotal);
  await expect(metricValue(page, 'baseline-total')).toHaveText(baselineTotal);
  const before = await page.evaluate(() => window.__pairHistory.length);
  await page.getByRole('button', { name: 'Swap', exact: true }).click();
  await expectPair(page, 'fictional-develop-00', 'fictional-develop-08');
  const swapped = await bars(page);
  for (const point of swapped) {
    const previous = original.find((entry) => entry.id === point.id);
    expect(point.candidate).toBe(previous.baseline);
    expect(point.baseline).toBe(previous.candidate);
    expect(point.delta).toBeCloseTo((previous.baseline / previous.candidate - 1) * 100, 8);
  }
  await expect(metricValue(page, 'candidate-total')).toHaveText(baselineTotal);
  await expect(metricValue(page, 'baseline-total')).toHaveText(candidateTotal);
  await expect(metricValue(page, 'not-comparable')).toHaveText('3');
  const added = exclusions.getByRole('row', { name: /Fictional added workload D/ });
  await expect(added.locator('td').nth(3)).toHaveText('Failed');
  await expect(added.locator('td').nth(4)).toHaveText('Unavailable in catalog');
  await expect.poll(() => page.evaluate(() => window.__pairHistory.length)).toBe(before + 1);
  const transitions = await page.evaluate((start) => window.__pairHistory.slice(start), before);
  expect(transitions.map((url) => {
    const params = new URL(url).searchParams;
    return [params.get('compareCandidate'), params.get('compareBaseline')];
  })).toEqual([['fictional-develop-00', 'fictional-develop-08']]);
  await page.goBack();
  await expectPair(page, 'fictional-develop-08', 'fictional-develop-00');
  expect(await bars(page)).toEqual(original);
  await page.goForward();
  await expectPair(page, 'fictional-develop-00', 'fictional-develop-08');
  expect(await bars(page)).toEqual(swapped);
  expect(await page.evaluate(() => window.__pairHistory.length)).toBe(before + 1);
  expect(new URL(page.url()).searchParams.get('campaign')).toBe('kept');
  expect(new URL(page.url()).hash).toBe('#pair');
  await page.getByRole('button', { name: 'Swap', exact: true }).click();
  expect(await bars(page)).toEqual(original);
  await selectRun(page, 'Candidate run', 'fictional-branch-02');
  await expectPair(page, 'fictional-branch-02', 'fictional-develop-00');
  const baseline = await page.getByRole('combobox', { name: 'Baseline run' }).inputValue();
  await page.getByRole('combobox', { name: 'Candidate run' }).fill('no-such-attempt');
  await expect(page.getByRole('option')).toHaveCount(0);
  await expect(page.getByText('No runs match this search')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('combobox', { name: 'Baseline run' })).toHaveValue(baseline);
});

test('branch-only publications keep comparison filters usable and long linked identities exact', async ({ page }) => {
  const publication = createSchema2Publication();
  publication.runs = publication.runs.filter((run) => ['fictional-branch-01', 'fictional-branch-04'].includes(run.id));
  const [candidate, baseline] = publication.runs;
  candidate.id = `fictional-candidate-${'x'.repeat(220)}`;
  baseline.id = `fictional-baseline-${'y'.repeat(220)}`;
  for (const run of publication.runs) run.comparisonId = run.id;
  publication.index.runFiles = publication.runs.map((run) => `runs/${run.id}.json`);
  // Valid fictional HTTP publication, with no develop attempts or disk fixture.
  await installPublication(page, { publication });
  const query = new URLSearchParams({ view: 'compare', compareCandidate: candidate.id, compareBaseline: baseline.id, targets: 'gfx1250', modes: 'ST' });
  await openDashboard(page, `/?${query}`);
  await expect(page.getByTestId('dashboard-route-error')).toHaveCount(0);
  await expectPair(page, candidate.id, baseline.id);
  await expect(metricValue(page, 'comparable')).toHaveText('4');
  await expect(chartFor(page)).toBeVisible();
  for (const [name, remaining] of [['gfx1250', '0'], ['ST', '0'], ['Triton', '2'], ['Llama', '2']]) {
    const control = page.getByRole('checkbox', { name, exact: true });
    await expect(control).toBeEnabled();
    await expect(control).toBeChecked();
    await control.uncheck();
    await expect(metricValue(page, 'comparable')).toHaveText(remaining);
    if (remaining === '0') await expect(chartFor(page)).toHaveCount(0);
    await control.check();
    await expect(metricValue(page, 'comparable')).toHaveText('4');
    await expect(chartFor(page)).toBeVisible();
  }
  await page.reload();
  await ready(page);
  await expect(page.getByTestId('dashboard-route-error')).toHaveCount(0);
  await expectPair(page, candidate.id, baseline.id);
  await expect(metricValue(page, 'comparable')).toHaveText('4');
  // Branch measurements must not become canonical Overview measurements.
  await page.getByRole('tab', { name: 'Overview', exact: true }).click();
  await expect(page.getByTestId('metric-card-total-duration')).toContainText('—');
  await page.getByRole('tab', { name: 'Run Comparison', exact: true }).click();
  await expectPair(page, candidate.id, baseline.id);
  await expect(page.getByRole('checkbox', { name: 'ST', exact: true })).toBeEnabled();
  await expect(metricValue(page, 'comparable')).toHaveText('4');
});

test('zero baselines remain measured, arbitrary metadata stays aligned and chart text is escaped', async ({ page }) => {
  const publication = createSchema2Publication();
  const injected = '<img src=x onerror="window.__tooltipInjection=true">';
  for (const catalog of Object.values(publication.catalogs)) catalog.tests.find((test) => test.id === 'b').name = injected;
  for (const [id, side] of [['fictional-develop-14', 'baseline'], ['fictional-branch-01', 'candidate']]) {
    publication.runs.find((run) => run.id === id).environment.push(
      { key: 'custom', label: 'Custom setting', value: side },
      { key: 'same', label: 'Unchanged setting', value: false },
      { key: `${side}Only`, label: `${side} only`, value: 0 },
    );
  }
  await installPublication(page, { publication });
  await openDashboard(page, '/?view=compare&compareCandidate=fictional-branch-01&compareBaseline=fictional-develop-14&targets=gfx1250&modes=ST');
  await expect(metricValue(page, 'comparable')).toHaveText('4');
  await expect(page.getByText('1 percentage unavailable', { exact: true })).toBeVisible();
  await page.getByText(/^View comparable results/).click();
  const zero = page.getByRole('table', { name: 'Comparable benchmark results' }).getByRole('row', { name: /Fictional GEMM/ });
  await expect(zero.getByRole('cell').nth(1)).toHaveText(formatDuration(0));
  await expect(zero).toContainText('percentage unavailable');
  expect((await bars(page)).map((point) => point.id)).not.toContain('gfx1250:ST:a');
  const metadata = page.getByTestId('metadata-row-custom');
  await expect(metadata.getByRole('cell').nth(0)).toHaveText('baseline');
  await expect(metadata.getByRole('cell').nth(1)).toHaveText('candidate');
  await expect(metadata).toHaveAttribute('data-different', 'true');
  await expect(metadata).not.toContainText('Changed');
  await expect(page.getByTestId('metadata-row-same')).toHaveAttribute('data-different', 'false');
  await expect(page.getByTestId('metadata-row-baselineOnly').getByRole('cell').nth(1)).toHaveText('Not provided');
  await expect(page.getByTestId('metadata-row-candidateOnly').getByRole('cell').nth(0)).toHaveText('Not provided');
  const html = await readChart(chartFor(page), (instance) => {
    const option = instance.getOption();
    const dataIndex = option.series[0].data.findIndex((point) => point.comparison.candidateTest.logicalTestId === 'b');
    instance.dispatchAction({ type: 'showTip', seriesIndex: 0, dataIndex });
    return option.tooltip[0].formatter({ data: option.series[0].data[dataIndex] });
  });
  expect(html).toContain('&lt;img');
  expect(html).not.toContain('<img');
  await expect(page.locator('img[src="x"]')).toHaveCount(0);
  expect(await page.evaluate(() => window.__tooltipInjection)).toBeUndefined();
});
