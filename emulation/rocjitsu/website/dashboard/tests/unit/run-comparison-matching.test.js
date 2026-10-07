import { expect, test } from 'vitest';
import { compareRuns, selectRunComparison } from '../../src/data/selectors.js';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';
const data = validatePublishedDashboardData(createSchema2Publication()).data;
const filters = { targets: ['gfx1250'], suites: data.suites, modes: ['ST'] };
const older = data.runs.find(({ catalogId }) => catalogId === 'fictional-old');
const current = data.latestCommitRun;

test('candidate and baseline nullability preserve unavailable totals', () => {
  expect(compareRuns(null, current, filters)).toEqual([]);
  expect(compareRuns(current, null, filters).every(({ comparable, baselineTest, delta }) => !comparable && !baselineTest && delta === null)).toBe(true);
  expect(selectRunComparison(null, null, filters)).toMatchObject({ comparable: [], notComparable: [], candidateDuration: null, baselineDuration: null, aggregateDelta: null });
});

test('catalog-only exclusions are symmetric when swapping exact target/mode pairs', () => {
  const forward = selectRunComparison(current, older, filters);
  const reverse = selectRunComparison(older, current, filters);
  expect(forward.comparable).toHaveLength(2);
  expect(reverse.comparable).toHaveLength(2);
  expect(forward.notComparable).toHaveLength(3);
  expect(reverse.notComparable).toHaveLength(3);
  expect(forward.candidateDuration).toBe(reverse.baselineDuration);
  expect(forward.baselineDuration).toBe(reverse.candidateDuration);
  expect(forward.comparable.every(({ candidateTest, baselineTest }) => candidateTest.testId === baselineTest.testId)).toBe(true);
});

test('failed results are exclusions; counts and sums use only matched completed rows', () => {
  const failure = data.runs.find(({ runId }) => runId === 'fictional-develop-09');
  const result = selectRunComparison(failure, current, filters);
  expect(result.notComparable).toHaveLength(1);
  expect(result.notComparable[0].candidateTest.status).toBe('failed');
  expect(result.comparable).toHaveLength(3);
  expect(result.counts.faster + result.counts.slower + result.counts.neutral + result.counts.unavailable).toBe(3);
  expect(result.candidateDuration).toBeCloseTo(result.comparable.reduce((total, { candidateTest }) => total + candidateTest.durationSeconds, 0));
});
