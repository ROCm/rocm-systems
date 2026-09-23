import { describe, expect, test } from 'vitest';
import { compareRuns, selectRunComparison } from '../../src/data/selectors.js';
import { commitShaFor } from '../../src/data/runOrdering.js';
import { benchmarkData } from '../fixtures/publishedData.js';

const filters = { targets: ['gfx1250'], suites: benchmarkData.suites };

// Two real gfx1250 runs to use as candidate/baseline in the "both present" case.
const gfx1250Runs = benchmarkData.runs.filter((run) =>
  run.tests.some((result) => result.target === 'gfx1250'),
);
const baselineRun = gfx1250Runs[0];
const candidateRun = gfx1250Runs.at(-1);

describe('compareRuns across candidate/baseline nullability', () => {
  // candidate = null → guard clause returns [] before touching the baseline.
  test('no candidate → empty array', () => {
    expect(compareRuns(null, baselineRun, filters)).toEqual([]);
  });

  // baseline = null, candidate present → rows are produced, but none are comparable.
  test('no baseline → all rows are not comparable', () => {
    const rows = compareRuns(candidateRun, null, filters);

    expect(rows).not.toHaveLength(0);
    expect(rows.every((row) => row.comparable === false)).toBe(true);
    expect(rows.every((row) => row.baselineTest === null && row.delta === null)).toBe(true);
  });

  // both null → the candidate half of the guard still returns [].
  test('both null → empty array', () => {
    expect(compareRuns(null, null, filters)).toEqual([]);
  });

  // both present → the real comparison path.
  test('candidate and baseline present → comparison rows', () => {
    const rows = compareRuns(candidateRun, baselineRun, filters);

    expect(rows).not.toHaveLength(0);
    // Every row's `comparable` flag agrees with its test data and delta.
    for (const row of rows) {
      if (row.comparable) {
        expect(typeof row.delta).toBe('number');
        expect(Number.isFinite(row.delta)).toBe(true);
      } else {
        expect(row.delta).toBeNull();
      }
    }
    // At least one pair actually compared on this path.
    expect(rows.some((row) => row.comparable)).toBe(true);
  });
});

test('run comparison counts catalog-only tests symmetrically when runs are swapped', () => {
  const runForCommit = (prefix) => benchmarkData.runs
    .find((run) => commitShaFor(run).startsWith(prefix));
  const catalogV2Run = runForCommit('0aa037d9');
  const catalogV1Run = runForCommit('784750dd');

  const forward = selectRunComparison(catalogV2Run, catalogV1Run, filters);
  const reverse = selectRunComparison(catalogV1Run, catalogV2Run, filters);

  expect(forward.comparable).toHaveLength(5);
  expect(forward.notComparable).toHaveLength(2);
  expect(forward.notComparable.every(
    (item) => item.candidateTest !== null && item.baselineTest === null,
  )).toBe(true);

  expect(reverse.comparable).toHaveLength(5);
  expect(reverse.notComparable).toHaveLength(2);
  expect(reverse.notComparable.every(
    (item) => item.candidateTest === null && item.baselineTest !== null,
  )).toBe(true);
});

test('run comparison marks rows without completed benchmark data as not comparable', () => {
  const runForCommit = (prefix) => benchmarkData.runs
    .find((run) => commitShaFor(run).startsWith(prefix));
  const august29 = runForCommit('19872076');
  const august28 = runForCommit('87c0b32c');
  const august27 = runForCommit('0db03af1');

  const candidateIssue = selectRunComparison(august29, august28, filters).notComparable.find(
    (item) => item.candidateTest?.logicalTestId === 'triton-gemm-f16-1024',
  );
  expect(candidateIssue).toMatchObject({ comparable: false, delta: null });
  expect(candidateIssue.candidateTest.status).not.toBe('completed');
  expect(candidateIssue.baselineTest.status).toBe('completed');

  const bothIssue = selectRunComparison(august29, august28, filters).notComparable.find(
    (item) => item.candidateTest?.logicalTestId === 'triton-softmax-f32-4096',
  );
  expect(bothIssue).toMatchObject({ comparable: false, delta: null });
  expect(bothIssue.candidateTest.status).not.toBe('completed');
  expect(bothIssue.baselineTest.status).not.toBe('completed');

  const baselineIssue = selectRunComparison(august28, august27, filters).notComparable.find(
    (item) => item.candidateTest?.logicalTestId === 'tensile-gemm-fp8-4096',
  );
  expect(baselineIssue).toMatchObject({ comparable: false, delta: null });
  expect(baselineIssue.candidateTest.status).toBe('completed');
  expect(baselineIssue.baselineTest.status).not.toBe('completed');
});
