import { expect, test } from 'vitest';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';

test.each([null, undefined, '123', {}, [0], [-1], [NaN], [Infinity], [true], ['2'], []])('unusable completed timings %j retain an explicit unavailable row', (samples) => {
  const source = createMaintainedPublication();
  source.runs[0].benchmark_results[0].timing_results_s = samples;
  const row = adaptPublishedDashboardData(source).data.runs[0].tests[0];
  expect(row.durationSeconds).toBeNull();
  expect(row.unavailableReason).toBe('No usable completed timing samples');
});

test.each([[4503599627370495], [4503599627370495, 4503599627370495], [1.25], [1, 1.5], [1, 1.25, 2]])('completed median retains the supported numeric policy for %j', (...samples) => {
  const source = createMaintainedPublication();
  source.runs[0].benchmark_results[0].timing_results_s = samples;
  const value = adaptPublishedDashboardData(source).data.runs[0].tests[0].durationSeconds;
  const expected = samples.length === 1 ? samples[0] : samples.length === 2 ? (samples[0] + samples[1]) / 2 : 1.25;
  expect(value).toBe(expected);
});
