import { describe, expect, test } from 'vitest';
import { compareRuns } from '../../src/data/selectors.js';
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

  // baseline = null, candidate present → rows are produced, but none are
  // comparable: every one is flagged 'Missing from baseline'.
  test('no baseline → rows all marked missing from baseline', () => {
    const rows = compareRuns(candidateRun, null, filters);

    expect(rows).not.toHaveLength(0);
    expect(rows.every((row) => row.comparable === false)).toBe(true);
    expect(rows.every((row) => row.notComparableReason === 'Missing from baseline')).toBe(true);
  });

  // both null → the candidate half of the guard still returns [].
  test('both null → empty array', () => {
    expect(compareRuns(null, null, filters)).toEqual([]);
  });

  // both present → the real comparison path.
  test('candidate and baseline present → comparison rows', () => {
    const rows = compareRuns(candidateRun, baselineRun, filters);

    expect(rows).not.toHaveLength(0);
    // Every row's `comparable` flag agrees with its delta/reason fields.
    for (const row of rows) {
      if (row.comparable) {
        expect(typeof row.delta).toBe('number');
        expect(Number.isFinite(row.delta)).toBe(true);
        expect(row.notComparableReason).toBeNull();
      } else {
        expect(row.delta).toBeNull();
        expect(row.notComparableReason).toEqual(expect.any(String));
      }
    }
    // At least one pair actually compared on this path.
    expect(rows.some((row) => row.comparable)).toBe(true);
  });
});
