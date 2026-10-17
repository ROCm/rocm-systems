import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { alpha, ThemeProvider } from '@mui/material/styles';
import { describe, expect, test, vi } from 'vitest';
import BenchmarkHistoryChart from '../../src/components/benchmarks/BenchmarkHistoryChart.jsx';
import { createDashboardTheme } from '../../src/theme/theme.js';
import { categoryColors } from '../../src/theme/tokens.js';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { selectBenchmarkSeries } from '../../src/data/selectors.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

const captured = vi.hoisted(() => ({ chart: null }));
vi.mock('../../src/components/shared/Chart.jsx', () => ({ default: (props) => {
  captured.chart = props;
  return createElement('div');
} }));

function validatedTarget(target) {
  const publication = createSchema2Publication();
  for (const catalog of Object.values(publication.catalogs)) {
    catalog.configurations = Object.fromEntries(Object.entries(catalog.configurations).map(([key, tests]) => [key.replace('gfx1250:', `${target}:`), tests]));
  }
  for (const run of publication.runs) {
    for (const configuration of run.configurations) if (configuration.target === 'gfx1250') configuration.target = target;
  }
  const { data } = validatePublishedDashboardData(publication);
  expect(data.targets).toContain(target);
  return data;
}

describe.each(['light', 'dark'])('%s benchmark history category colors', (mode) => {
  test.each(['constructor', 'toString', '__proto__', 'hasOwnProperty', 'unknown-target', 'gfx1250'])('uses only recognized palette entries for validated target %s', (target) => {
    const data = validatedTarget(target);
    const filters = { targets: [target, 'gfx950'], suites: ['Triton', 'Llama'], modes: ['ST', 'MT'] };
    const viewModel = selectBenchmarkSeries(data, filters, 'a');
    expect(viewModel.series.some((series) => series.target === target)).toBe(true);
    renderToStaticMarkup(createElement(ThemeProvider, { theme: createDashboardTheme(mode) }, createElement(BenchmarkHistoryChart, {
      data, filters, benchmark: { id: 'a', name: 'Fictional GEMM' },
    })));
    const option = captured.chart.option;
    const expectedColors = viewModel.series.map((series) => ['gfx1250', 'gfx950'].includes(series.target) ? categoryColors[mode][series.target] : series.color);
    expect(option.color).toEqual(expectedColors);
    const lines = option.series.filter((series) => series.type === 'line');
    const selections = option.series.filter((series) => series.name.endsWith(' selected points'));
    for (const [index, color] of expectedColors.entries()) {
      expect(typeof color).toBe('string');
      expect(lines[index].lineStyle.color).toBe(color);
      expect(lines[index].lineStyle.shadowColor).toBe(alpha(color, 0.22));
      expect(lines[index].itemStyle.color).toBe(color);
      expect(selections[index].itemStyle.color).toBe(color);
    }
  });
});
