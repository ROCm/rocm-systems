import { expect, test } from 'vitest';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { selectBenchmarkSeries } from '../../src/data/selectors.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';

function publication() {
  const source = createMaintainedPublication();
  for (const run of source.runs) for (const row of run.benchmark_results) row.operation = 'Matrix multiplication followed by reduction';
  return source;
}

test('preserves literal workload IDs, operation, ordered samples and sparse target/policy configurations', () => {
  const source = publication();
  const run = source.runs[12];
  for (const row of run.benchmark_results) row.timing_results_s = [0.03, 0.01, 0.02];
  const before = structuredClone(source);
  const { data, sourceData } = adaptPublishedDashboardData(source);
  const normalized = data.runs.find(({ runId }) => runId === run.publication.id);
  expect(normalized.configurations).toHaveLength(3);
  expect(normalized.tests.every((result) => result.operation === 'Matrix multiplication followed by reduction'
    && result.logicalTestId === result.id && result.testId === `${result.target}/${result.mode}:${result.id}`
    && result.timing_results_s.join(',') === '0.03,0.01,0.02' && result.durationSeconds === 0.02)).toBe(true);
  const series = selectBenchmarkSeries(data, { targets: ['gfx950'], suites: data.suites, modes: ['single', 'default'] }, 'a');
  expect(series.series.map(({ key }) => key)).toEqual(['gfx950/single', 'gfx950/default']);
  expect(series.series[0].points[12].value).toBe(0.02);
  expect(series.series[1].points[12]).toBeNull();
  expect(sourceData.runs).toEqual(before.runs);
  expect(source).toEqual(before);
});
