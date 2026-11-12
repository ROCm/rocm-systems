// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { THREAD_POLICIES } from './runSchema.mjs';

export const DURATION_CHANGE_BAND_PERCENT = 3;

export function classifyDurationChange(value, tolerance = DURATION_CHANGE_BAND_PERCENT) {
  if (!Number.isFinite(value)) return 'unavailable';
  if (value > tolerance) return 'slower';
  if (value < -tolerance) return 'faster';
  return 'neutral';
}

const modesFor = (filters) => filters.modes ?? THREAD_POLICIES;
export const completed = (test) => test?.status === 'completed' && Number.isFinite(test.durationSeconds);
const sum = (tests) => tests.reduce((total, test) => total + test.durationSeconds, 0);
const round = (value) => Number(value.toPrecision(15));
const percentage = (candidate, baseline) => {
  const value = Number.isFinite(candidate) && Number.isFinite(baseline) && baseline > 0 ? (candidate - baseline) / baseline * 100 : null;
  return Number.isFinite(value) ? value : null;
};
const resultMap = (run) => new Map((run?.tests ?? []).map((test) => [test.testId, test]));
export const testMatches = (test, filters) => filters.targets.includes(test.target) && filters.suites.includes(test.suite) && modesFor(filters).includes(test.mode);
const selected = (run, filters) => (run?.tests ?? []).filter((test) => testMatches(test, filters));
export function compareRuns(candidate, baseline, filters) {
  const baselineTests = resultMap(baseline);
  return selected(candidate, filters).map((candidateTest) => {
    const baselineTest = baselineTests.get(candidateTest.testId) ?? null;
    const comparable = completed(candidateTest) && completed(baselineTest);
    return { candidateTest, baselineTest, comparable,
      delta: comparable ? percentage(candidateTest.durationSeconds, baselineTest.durationSeconds) : null,
      deltaSeconds: comparable ? candidateTest.durationSeconds - baselineTest.durationSeconds : null };
  });
}

function runSummary(run, filters) {
  const tests = selected(run, filters); const completedTests = tests.filter(completed);
  return { total: tests.length, completed: completedTests.length, failed: tests.filter((t) => t.status === 'failed').length,
    timeout: tests.filter((t) => t.status === 'timeout').length,
    duration: tests.length > 0 && completedTests.length === tests.length ? round(sum(completedTests)) : null,
    completionPercent: tests.length ? completedTests.length / tests.length * 100 : null };
}

// The table needs coverage and duration, not historical baseline comparisons.
export function selectRecentRunSummaries(runs, filters) {
  return runs.map((run) => ({ run, ...runSummary(run, filters) }));
}

export function selectRunComparison(candidate, baseline, filters, tolerance = DURATION_CHANGE_BAND_PERCENT) {
  const candidates = compareRuns(candidate, baseline, filters); const ids = new Set(candidates.map(({ candidateTest }) => candidateTest.testId));
  const baselineOnly = selected(baseline, filters).filter(({ testId }) => !ids.has(testId)).map((baselineTest) => ({ candidateTest: null, baselineTest, comparable: false, delta: null, deltaSeconds: null }));
  const comparisons = [...candidates, ...baselineOnly]; const comparable = comparisons.filter(({ comparable }) => comparable);
  const baselineDuration = comparable.length ? round(sum(comparable.map(({ baselineTest }) => baselineTest))) : null;
  const candidateDuration = comparable.length ? round(sum(comparable.map(({ candidateTest }) => candidateTest))) : null;
  const counts = { faster: 0, slower: 0, neutral: 0, unavailable: 0 };
  for (const { delta } of comparable) counts[classifyDurationChange(delta, tolerance)] += 1;
  return { comparable: comparable.sort((a, b) => (Number.isFinite(b.delta) ? Math.abs(b.delta) : -1) - (Number.isFinite(a.delta) ? Math.abs(a.delta) : -1)),
    notComparable: comparisons.filter(({ comparable }) => !comparable), baselineDuration, candidateDuration,
    aggregateDelta: percentage(candidateDuration, baselineDuration), counts, tolerance };
}
