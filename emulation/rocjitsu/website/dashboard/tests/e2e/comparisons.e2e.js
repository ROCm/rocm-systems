import { fixtureRunPath } from '../fixtures/runPath.js';
import { expect, test } from '@playwright/test';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { formatDuration, formatPercent } from '../../src/utils/formatters.js';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { readChart } from './helpers/chart.js';
import { readMetricGeometry, metricContainmentFailures } from './helpers/metricGeometry.js';
import { selectRunComparison } from '../../src/data/selectors.js';
import { installPublication, openDashboard, ready, selectRun } from './helpers/dashboard.js';

const chartFor = (page) => page.getByRole('img', { name: 'Performance change by benchmark comparison chart' });
const bars = (page) => readChart(chartFor(page), (instance) => instance.getOption().series[0].data.map((point) => ({
  id: point.comparison.candidateTest.testId, delta: point.value,
  candidate: point.comparison.candidateTest.durationSeconds,
  baseline: point.comparison.baselineTest.durationSeconds,
})));
for (const width of [1440, 390]) test(`numeric availability stays truthful and ordinary charts restore at ${width}px`, async ({ page }, testInfo) => {
  await page.setViewportSize({ width, height: 900 });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const publication = createMaintainedPublication();
  const scenarios = [
    { id: 'fictional-develop-14', seconds: 1e-300, delta: null },
    { id: 'fictional-develop-15', seconds: 2.5e-291, delta: 1.6e308 },
    { id: 'fictional-develop-16', seconds: 5e-291, delta: 8e307 },
    { id: 'fictional-develop-17', seconds: 1e-289, delta: 3.9999999999999994e306 },
  ];
  for (const { id, seconds } of [...scenarios, { id: 'fictional-branch-01', seconds: 4e15 }]) {
    for (const result of publication.runs.find((run) => run.publication.id === id).benchmark_results) {
       Object.assign(result, { status: 'completed', error: null, timing_results_s: [seconds] });
    }
  }
  adaptPublishedDashboardData(publication);
  await installPublication(page, { publication });
  const branch = publication.runs.find((run) => run.publication.id === 'fictional-branch-01').publication.source.branch;
  await openDashboard(page, `/?${new URLSearchParams({ view: 'branch', branch, run: 'fictional-branch-01', reference: 'fictional-develop-14', manual: '1', target: 'gfx1250', mode: 'single', detail: '1' })}`);
  const card = page.getByTestId('branch-config-gfx1250-single');
  await expect(card).toContainText('Percentage unavailable');
  await expect(card).not.toContainText('zero reference runtime');
  await expect(card).toContainText('4 matched');
  const result = page.getByTestId(`${width < 768 ? 'branch-mobile-result-' : 'branch-result-'}gfx1250/single:a`);
  await expect(result).toBeVisible();
  await expect(result).toContainText(formatDuration(1e-300));
  await expect(result).toContainText(formatDuration(4e15));
  await expect(result).toContainText('Percentage unavailable');
  await expect(result).not.toContainText('zero reference runtime');
  await expect(result).toContainText('more time');
  const comparisonUrl = (baseline, candidate = 'fictional-branch-01') => `/?${new URLSearchParams({ view: 'compare', compareBaseline: baseline, compareCandidate: candidate, targets: 'gfx1250', modes: 'single' })}`;
  for (const scenario of scenarios) {
    await openDashboard(page, comparisonUrl(scenario.id));
    await expectPair(page, 'fictional-branch-01', scenario.id);
    await expect(metricValue(page, 'comparable')).toHaveText('4');
    if (scenario.id === 'fictional-develop-17') {
      await expect(chartFor(page)).toBeVisible();
      const domain = await readChart(chartFor(page), (instance) => ({
        extent: instance.getModel().getComponent('xAxis').axis.scale.getExtent(),
        blank: instance.getModel().getComponent('xAxis').axis.scale.isBlank(),
      }));
      expect(domain.blank).toBe(false);
      expect(domain.extent.every(Number.isFinite)).toBe(true);
      expect(Number.isFinite(domain.extent[1] - domain.extent[0])).toBe(true);
      expect(domain.extent[0]).toBeLessThan(domain.extent[1]);
      const plotted = await bars(page);
      expect(plotted).toHaveLength(4);
      for (const point of plotted) {
        expect(point.delta).toBe(scenario.delta);
        expect(point.delta).toBeGreaterThanOrEqual(domain.extent[0]);
        expect(point.delta).toBeLessThanOrEqual(domain.extent[1]);
      }
    } else {
      await expect(chartFor(page)).toHaveCount(0);
      await expect(page.getByText(scenario.delta === null
        ? 'No percentage changes available. Measured results remain in the results table below.'
        : 'Chart unavailable: percentage changes exceed the supported chart range. Measured results remain in the results table below.', { exact: true })).toBeVisible();
    }
    await page.getByText('View comparable results (4)', { exact: true }).click();
    const rows = page.getByRole('table', { name: 'Comparable benchmark results' }).locator('tbody tr');
    await expect(rows).toHaveCount(4);
    for (const row of await rows.all()) {
      await expect(row.getByRole('cell').nth(1)).toHaveText(formatDuration(scenario.seconds));
      await expect(row.getByRole('cell').nth(2)).toHaveText(formatDuration(4e15));
      await expect(row.getByRole('cell').nth(3)).toContainText(scenario.delta === null ? 'percentage unavailable' : formatPercent(scenario.delta));
    }
    await testInfo.attach(`numeric-${scenario.id}-${width}`, { body: await page.screenshot(), contentType: 'image/png' });
  }
  await openDashboard(page, comparisonUrl('fictional-branch-01', 'fictional-develop-15'));
  expect((await bars(page)).every((point) => point.delta === -100)).toBe(true);
  await openDashboard(page, comparisonUrl('fictional-develop-00', 'fictional-develop-08'));
  const ordinary = await bars(page);
  expect(ordinary).toHaveLength(2);
  expect(ordinary.every((point) => Number.isFinite(point.delta) && point.delta > -100 && point.delta < 100)).toBe(true);
  await expect(page.getByText(/^Chart unavailable:/)).toHaveCount(0);
  expect(errors).toEqual([]);
});

for (const width of [1440, 900, 390]) test(`comparison metric painted values remain contained at ${width}px`, async ({ page }, testInfo) => {
  test.setTimeout(60_000);
  await page.setViewportSize({ width, height: 900 });
  const publication = createMaintainedPublication();
  for (const [id, seconds] of [['fictional-develop-17', 1e-289], ['fictional-develop-16', 2e15], ['fictional-branch-01', 4e15]]) {
    for (const result of publication.runs.find((run) => run.publication.id === id).benchmark_results) {
       Object.assign(result, { status: 'completed', error: null, timing_results_s: [seconds] });
    }
  }
  const data = adaptPublishedDashboardData(publication).data;
  await installPublication(page, { publication });
  const scenarios = [
    ['extreme', 'fictional-develop-17', 'fictional-branch-01'],
    ['long-totals', 'fictional-develop-16', 'fictional-branch-01'],
    ['reverse', 'fictional-branch-01', 'fictional-develop-17'],
    ['ordinary', 'fictional-develop-00', 'fictional-develop-08'],
    ['unavailable', 'fictional-missing', 'fictional-branch-01'],
  ];
  for (const theme of ['light', 'dark']) for (const [scenario, baselineId, candidateId] of scenarios) {
    await openDashboard(page, `/?${new URLSearchParams({ view: 'compare', compareBaseline: baselineId, compareCandidate: candidateId, targets: 'gfx1250', modes: 'single' })}`);
    const toggle = page.getByRole('button', { name: theme === 'dark' ? 'Use dark theme' : 'Use light theme', exact: true });
    if (await toggle.count()) await toggle.click();
    const baseline = data.allRuns.find((run) => run.runId === baselineId);
    const candidate = data.allRuns.find((run) => run.runId === candidateId);
    const model = selectRunComparison(candidate, baseline, { targets: ['gfx1250'], modes: ['single'], suites: ['Triton', 'Llama'] });
    const hasPair = Boolean(candidate && baseline);
    const expected = [formatPercent(model.aggregateDelta), model.comparable.length ? formatDuration(model.candidateDuration) : '—', model.comparable.length ? formatDuration(model.baselineDuration) : '—', hasPair ? String(model.comparable.length) : '—', hasPair ? String(model.notComparable.length) : '—'];
    const geometry = await readMetricGeometry(page);
    expect(geometry.map((cell) => cell.text)).toEqual(expected);
    await testInfo.attach(`metrics-${width}-${theme}-${scenario}`, { body: JSON.stringify(geometry, null, 2), contentType: 'application/json' });
    await testInfo.attach(`metrics-${width}-${theme}-${scenario}-scrolled`, { body: await page.screenshot(), contentType: 'image/png' });
    expect(metricContainmentFailures(geometry), 'painted values must fit their own content bounds without adjacent/caption/commit collision').toEqual([]);
    if (scenario === 'ordinary') expect(geometry.every((cell) => new Set(cell.value.map((line) => line.top)).size === 1)).toBe(true);
    const commits = page.getByTestId('comparison-metric-aggregate-change').getByTestId('compared-commits');
    if (hasPair) await expect(commits).toBeVisible();
    else await expect(commits).toHaveCount(0);
    if (scenario === 'extreme' && width === 1440 && theme === 'light') {
      const value = metricValue(page, 'aggregate-change');
      const saved = await value.evaluate((element) => {
        const number = element.querySelector('[data-metric-number]');
        const originals = [element, number].map((node) => node.getAttribute('style'));
        element.style.overflowWrap = 'normal';
        number.style.minWidth = 'auto';
        number.style.overflowWrap = 'normal';
        return originals;
      });
      try {
        const negative = await readMetricGeometry(page);
        expect(negative.map((cell) => cell.text)).toEqual(expected);
        const failures = metricContainmentFailures(negative);
        expect(failures).toContain('comparison-metric-aggregate-change: value outside right');
        await testInfo.attach('metrics-isolated-negative', { body: JSON.stringify({ negative, failures }), contentType: 'application/json' });
      } finally {
        await value.evaluate((element, originals) => {
          [element, element.querySelector('[data-metric-number]')].forEach((node, index) => {
            if (originals[index] === null) node.removeAttribute('style');
            else node.setAttribute('style', originals[index]);
          });
        }, saved);
      }
      expect(await value.evaluate((element) => [element, element.querySelector('[data-metric-number]')].map((node) => node.getAttribute('style')))).toEqual(saved);
      const restored = await readMetricGeometry(page);
      expect(metricContainmentFailures(restored)).toEqual([]);
      expect(restored).toEqual(geometry);
      await testInfo.attach('metrics-exact-restoration', { body: JSON.stringify(restored), contentType: 'application/json' });
    }
  }
});

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
  await openDashboard(page, '/?view=compare&compareCandidate=fictional-develop-08&compareBaseline=fictional-develop-00&targets=gfx1250&modes=single&campaign=kept#pair');
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
  const publication = createMaintainedPublication();
  publication.runs = publication.runs.filter((run) => ['fictional-branch-01', 'fictional-branch-04'].includes(run.publication.id));
  const [candidate, baseline] = publication.runs;
  candidate.publication.id = `fictional-candidate-${'x'.repeat(220)}`;
  baseline.publication.id = `fictional-baseline-${'y'.repeat(220)}`;
  publication.index.runFiles = publication.runs.map(fixtureRunPath);
  // Valid fictional HTTP publication, with no develop attempts or disk fixture.
  await installPublication(page, { publication });
  const query = new URLSearchParams({ view: 'compare', compareCandidate: candidate.publication.id, compareBaseline: baseline.publication.id, targets: 'gfx1250', modes: 'single' });
  await openDashboard(page, `/?${query}`);
  await expect(page.getByTestId('dashboard-route-error')).toHaveCount(0);
  await expectPair(page, candidate.publication.id, baseline.publication.id);
  await expect(metricValue(page, 'comparable')).toHaveText('4');
  await expect(chartFor(page)).toBeVisible();
  for (const [name, remaining] of [['gfx1250', '0'], ['single', '0'], ['Triton', '2'], ['Llama', '2']]) {
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
  await expectPair(page, candidate.publication.id, baseline.publication.id);
  await expect(metricValue(page, 'comparable')).toHaveText('4');
  // Branch measurements must not become canonical Overview measurements.
  await page.getByRole('tab', { name: 'Overview', exact: true }).click();
  const duration = page.getByTestId('metric-card-latest-duration');
  await expect(page.getByRole('checkbox', { name: 'gfx1250', exact: true })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: 'single', exact: true })).toBeChecked();
  await expect(duration.locator('[data-configuration]')).toHaveCount(1);
  await expect(duration.locator('[data-configuration="gfx1250/single"]')).toContainText('No shared workloads');
  await expect(duration.locator('[data-configuration="gfx1250/single"] p').last()).toHaveText('No shared workloads');
  await expect(page.getByTestId('speedup-empty-state')).toBeVisible();
  await expect(page.getByTestId('performance-trend').getByRole('img')).toHaveCount(0);
  await expect(page.getByTestId('recent-runs-table').locator('tbody tr')).toHaveCount(0);
  await expect(page.getByTestId('largest-changes').locator('[data-change-state]')).toHaveCount(0);
  await page.getByRole('tab', { name: 'Run Comparison', exact: true }).click();
  await expectPair(page, candidate.publication.id, baseline.publication.id);
  await expect(page.getByRole('checkbox', { name: 'single', exact: true })).toBeEnabled();
  await expect(metricValue(page, 'comparable')).toHaveText('4');
});

test('positive overflow ratios remain measured, arbitrary metadata stays aligned and chart text is escaped', async ({ page }) => {
  const publication = createMaintainedPublication();
  const injected = '<img src=x onerror="window.__tooltipInjection=true">';
  for (const run of publication.runs) for (const row of run.benchmark_results) if (row.id === 'b') row.name = injected;
  for (const [id, side] of [['fictional-develop-14', 'baseline'], ['fictional-branch-01', 'candidate']]) {
    const run = publication.runs.find((run) => run.publication.id === id);
    Object.assign(run.provenance.auxiliary, { custom: side, same: false, [`${side}Only`]: 0 });
    run.benchmark_results.find((result) => result.target === 'gfx1250' && result.thread_policy === 'single' && result.id === 'a').timing_results_s = [side === 'baseline' ? 1e-300 : 4e15];
  }
  await installPublication(page, { publication });
  await openDashboard(page, '/?view=compare&compareCandidate=fictional-branch-01&compareBaseline=fictional-develop-14&targets=gfx1250&modes=single');
  await expect(metricValue(page, 'comparable')).toHaveText('4');
  await expect(page.getByText('1 percentage unavailable', { exact: true })).toBeVisible();
  await page.getByText(/^View comparable results/).click();
  const measured = page.getByRole('table', { name: 'Comparable benchmark results' }).getByRole('row', { name: /Fictional GEMM/ });
  await expect(measured.getByRole('cell').nth(1)).toHaveText(formatDuration(1e-300));
  await expect(measured.getByRole('cell').nth(2)).toHaveText(formatDuration(4e15));
  await expect(measured).toContainText('percentage unavailable');
  expect((await bars(page)).map((point) => point.id)).not.toContain('gfx1250/single:a');
  const metadata = page.getByTestId('metadata-row-auxiliary.custom');
  await expect(metadata.getByRole('cell').nth(0)).toHaveText('baseline');
  await expect(metadata.getByRole('cell').nth(1)).toHaveText('candidate');
  await expect(metadata).toHaveAttribute('data-different', 'true');
  await expect(metadata).not.toContainText('Changed');
  await expect(page.getByTestId('metadata-row-auxiliary.same')).toHaveAttribute('data-different', 'false');
  await expect(page.getByTestId('metadata-row-auxiliary.baselineOnly').getByRole('cell').nth(1)).toHaveText('—');
  await expect(page.getByTestId('metadata-row-auxiliary.candidateOnly').getByRole('cell').nth(0)).toHaveText('—');
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
