import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { themed } from '../helpers/themed.js';
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
  renderToStaticMarkup(themed(BenchmarkHistoryChart, {
    data,
    filters: { targets: ['gfx1250', 'gfx950'], suites: ['Triton', 'Llama'], modes: ['single', 'default'] },
    benchmark: { id: 'a', name: 'Fictional GEMM' },
  }));
  return captured.chart.option;
}

it('connects benchmark history lines across unavailable measurements without inventing points', () => {
  const lines = chartOptions().series.filter((series) => series.type === 'line');
  expect(lines.every((series) => series.connectNulls === true)).toBe(true);
  const mt = lines.find((series) => series.name === 'gfx1250 default');
  const values = mt.data.map((point) => point.value[1]);
  const measured = values.map((value, index) => value === null ? -1 : index).filter((index) => index >= 0);
  expect(values.slice(measured[0], measured.at(-1) + 1)).toContain(null);
  expect(values.every((value) => value === null || value > 0)).toBe(true);
});

it('benchmark tooltip reuses the performance-trend layout: commit, commit time, one row per series', () => {
  const option = chartOptions();
  const points = option.series.filter((series) => series.type === 'line').map((series, seriesIndex) => ({
    seriesType: 'line', seriesName: series.name, seriesIndex, color: '#000', data: series.data.find((point) => point.record),
  }));
  const html = option.tooltip.formatter(points);
  expect(option.tooltip.className).toBe('performance-trend-tooltip');
  expect(html).toContain('Commit ');
  expect(html).toContain('gfx1250 single');
  expect(html).toContain('gfx950 default');
  expect((html.match(/performance-trend-tooltip-row/g) ?? []).length).toBe(points.length);
  // Swatches use the series line color, never the (paper-colored) fill of hollow single-policy points.
  expect(html).not.toContain('stroke="#000"');
  for (const label of ['Commit name', 'Catalog', 'Execution time', 'Branch']) expect(html).not.toContain(`${label} ·`);
});

it('escapes compact tooltip labels and never describes missing measurements as zero', () => {
  const option = chartOptions();
  const point = option.series.find((series) => series.type === 'line').data.find((entry) => entry?.record);
  const html = option.tooltip.formatter([{ seriesType: 'line', seriesName: '<img src=x onerror=alert(1)>', seriesIndex: 0, color: '#000', data: point }]);
  expect(html).toContain('&lt;img');
  expect(html).not.toContain('<img');
  expect(option.tooltip.formatter([{ seriesType: 'line', data: null }])).toBe('No measured result for this benchmark');
});
