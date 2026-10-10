import { expect, test } from 'vitest';
import { validatePublishedResult, validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

const completed = (samples, durationSeconds = 2) => ({
  testId: 'example', status: 'completed', durationSeconds, error: null, timing_results_s: samples,
});

test.each([null, undefined, '123', {}, [0], [-1], [NaN], [Infinity], [true], ['2'], []].map((samples) => ({ samples })))(
  'rejects invalid completed timing samples $samples', ({ samples }) => {
    expect(() => validatePublishedResult(completed(samples))).toThrow(/timing_results_s/);
  },
);

test('rejects a duration that disagrees with the sample median', () => {
  expect(() => validatePublishedResult(completed([1, 9, 2], 9))).toThrow(/median/);
});

test.each([
  [[1, 9, 2], 2],
  [[9, 1, 4, 2], 3],
  [[4], 4],
])('accepts median duration without reordering samples %j', (samples, duration) => {
  const result = completed(samples, duration);
  const before = structuredClone(result);
  expect(validatePublishedResult(result)).toEqual(before);
});

test.each(['failed', 'timeout'])('preserves %s partial samples with null duration', (status) => {
  for (const timing_results_s of [[], [5, 2]]) {
    const result = { testId: 'example', status, durationSeconds: null, error: null, timing_results_s };
    expect(validatePublishedResult(result)).toEqual(result);
  }
  expect(() => validatePublishedResult({ testId: 'example', status, durationSeconds: null, error: null, timing_results_s: [0] })).toThrow(/timing_results_s/);
});

test('old publications retain their durations without invented samples', () => {
  const source = createSchema2Publication();
  const { data } = validatePublishedDashboardData(source);
  for (const run of data.allRuns) {
    for (const result of run.tests) expect(result).not.toHaveProperty('timing_results_s');
  }
  expect(data.runs.find(({ runId }) => runId === 'fictional-develop-14').tests[0].durationSeconds).toBe(0);
});
