import { expect, test } from '@playwright/test';
import { readChart } from './helpers/chart.js';
import { ensureGridBenchmark, openDashboard } from './helpers/dashboard.js';

for (const theme of ['light', 'dark']) for (const kind of ['benchmark', 'comparison']) {
  test(`${kind} tooltips avoid scaled compositor layers in ${theme} theme`, async ({ browser }) => {
    const context = await browser.newContext({
      viewport: { width: 1439, height: 643 }, screen: { width: 1835, height: 768 }, deviceScaleFactor: 1.875,
    });
    try {
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await openDashboard(page, kind === 'comparison'
        ? '/?view=compare&compareCandidate=fictional-develop-23&compareBaseline=fictional-develop-17'
        : '/?view=benchmarks');
      if (theme === 'dark') await page.getByRole('button', { name: 'Use dark theme', exact: true }).click();
      if (kind === 'benchmark') await ensureGridBenchmark(page, 'Fictional GEMM');
      const chart = page.getByRole('img', { name: kind === 'comparison'
        ? 'Performance change by benchmark comparison chart' : 'Fictional GEMM duration history', exact: true });
      await chart.scrollIntoViewIfNeeded();
      for (const last of [false, true]) {
        await readChart(chart, (instance, { kind, last }) => {
          const series = instance.getOption().series;
          const valid = (point) => kind === 'comparison' ? Number.isFinite(point?.value) : point?.record && Number.isFinite(point?.value);
          const seriesIndex = series.findIndex((row) => row.type === (kind === 'comparison' ? 'bar' : 'line') && row.data.some(valid));
          const dataIndex = last ? series[seriesIndex].data.findLastIndex(valid) : series[seriesIndex].data.findIndex(valid);
          instance.dispatchAction({ type: 'showTip', seriesIndex, dataIndex });
        }, { kind, last });
        const tooltip = chart.locator('.dashboard-chart-tooltip');
        await expect(tooltip).toBeVisible();
        await expect(tooltip).toContainText(kind === 'comparison' ? 'Candidate' : 'Commit');
        const position = await tooltip.evaluate((element) => {
          const style = getComputedStyle(element);
          const rect = element.getBoundingClientRect();
          const chart = element.parentElement.getBoundingClientRect();
          return { transform: style.transform, willChange: style.willChange, backdropFilter: style.backdropFilter,
            x: rect.x * devicePixelRatio, y: rect.y * devicePixelRatio,
            inside: rect.left >= chart.left - 1 && rect.right <= chart.right + 1 && rect.top >= chart.top - 1 && rect.bottom <= chart.bottom + 1 };
        });
        expect(position.transform).toBe('none');
        expect(position.willChange).toBe('auto');
        expect(position.backdropFilter).toBe('none');
        expect(Math.abs(position.x - Math.round(position.x))).toBeLessThan(0.04);
        expect(Math.abs(position.y - Math.round(position.y))).toBeLessThan(0.04);
        expect(position.inside).toBe(true);
      }
      expect(errors).toEqual([]);
    } finally {
      await context.close();
    }
  });
}
