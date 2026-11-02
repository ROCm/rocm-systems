import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it, vi } from 'vitest';
import BenchmarkHistoryChart from '../../src/components/benchmarks/BenchmarkHistoryChart.jsx';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { setFixtureConfigurations } from '../fixtures/configurationScope.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';

const captured = vi.hoisted(() => ({ chart: null }));
vi.mock('../../src/components/shared/Chart.jsx', () => ({ default: (props) => {
  captured.chart = props;
  return createElement('div');
} }));

function chartOptions() {
  const publication = createMaintainedPublication();
  // A valid partial publication: this interior attempt has no gfx1250 default result.
  const missing = publication.runs.find((run) => run.publication.id === 'fictional-develop-10');
  setFixtureConfigurations(publication, missing, missing.benchmark_results.filter((config) => config.target !== 'gfx1250' || config.thread_policy !== 'default'));
  const data = adaptPublishedDashboardData(publication).data;
  renderToStaticMarkup(createElement(BenchmarkHistoryChart, {
    data,
    filters: { targets: ['gfx1250', 'gfx950'], suites: ['Triton', 'Llama'], modes: ['single', 'default'] },
    benchmark: { id: 'a', name: 'Fictional GEMM' },
  }));
  return captured.chart.option;
}

it('keeps benchmark history discontinuous across unavailable measurements with positive measurements', () => {
  const lines = chartOptions().series.filter((series) => series.type === 'line');
  expect(lines.every((series) => series.connectNulls === false)).toBe(true);
  const mt = lines.find((series) => series.name === 'gfx1250 default');
  const measured = mt.data.map((point, index) => point === null ? -1 : index).filter((index) => index >= 0);
  expect(mt.data.slice(measured[0], measured.at(-1) + 1)).toContain(null);
  expect(lines.every((series) => series.data.every((point) => point === null || point.value > 0))).toBe(true);
});

it('keeps benchmark tooltip compact and leaves full metadata in result details', () => {
  const option = chartOptions();
  const points = option.series.filter((series) => series.type === 'line').map((series) => ({
    seriesType: 'line', seriesName: series.name, marker: '', data: series.data.find((point) => point?.record),
  }));
  const html = option.tooltip.formatter(points);
  expect(html).toContain('Commit ');
  expect(html).toContain('gfx1250 single');
  expect(html).toContain('gfx950 default');
  for (const label of ['Commit name', 'Catalog', 'Commit time', 'Execution time', 'Branch']) expect(html).not.toContain(`${label} ·`);
  expect((html.match(/<br\s*\/?>/g) ?? []).length).toBeLessThanOrEqual(points.length + 1);
  expect(option.tooltip.confine).toBe(true);
  expect(option.tooltip.textStyle.fontSize).toBeLessThanOrEqual(12);
  expect(option.tooltip.extraCssText).toContain('max-width:240px');
});

it('escapes compact tooltip labels and never describes missing measurements as zero', () => {
  const option = chartOptions();
  const point = option.series.find((series) => series.type === 'line').data.find((entry) => entry?.record);
  const html = option.tooltip.formatter([{ seriesType: 'line', seriesName: '<img src=x onerror=alert(1)>', marker: '', data: point }]);
  expect(html).toContain('&lt;img');
  expect(html).not.toContain('<img');
  expect(option.tooltip.formatter([{ seriesType: 'line', data: null }])).toBe('No measured result for this benchmark');
});
