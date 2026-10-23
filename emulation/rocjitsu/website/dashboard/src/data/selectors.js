import { configurationKey, THREAD_POLICIES } from './runSchema.js';
import { targetColor } from '../utils/chartColors.js';
import { classifyDurationChange, DURATION_CHANGE_BAND_PERCENT } from '../utils/performance.js';
import { commitShaFor, commitTimestampFor, compareRunExecution, hasCommitTimestamp, sortRunsByCommit } from './runOrdering.js';


const modesFor = (filters) => filters.modes ?? THREAD_POLICIES;
const completed = (test) => test?.status === 'completed' && Number.isFinite(test.durationSeconds);
const sum = (tests) => tests.reduce((total, test) => total + test.durationSeconds, 0);
const round = (value) => Number(value.toPrecision(15));
const percentage = (candidate, baseline) => {
  const value = Number.isFinite(candidate) && Number.isFinite(baseline) && baseline > 0 ? (candidate - baseline) / baseline * 100 : null;
  return Number.isFinite(value) ? value : null;
};
const resultMap = (run) => new Map((run?.tests ?? []).map((test) => [test.testId, test]));
const testMatches = (test, filters) => filters.targets.includes(test.target) && filters.suites.includes(test.suite) && modesFor(filters).includes(test.mode);
const selected = (run, filters) => (run?.tests ?? []).filter((test) => testMatches(test, filters));
const pairsFor = (filters) => filters.targets.flatMap((target) => modesFor(filters).map((mode) => ({ target, mode, key: configurationKey(target, mode) })));

export function periodKey(timestamp, period = 'weekly') {
  if (timestamp == null || !Number.isFinite(new Date(timestamp).getTime())) return null;
  const date = new Date(timestamp);
  if (period === 'monthly') return date.toISOString().split('T')[0].slice(0, -3);
  if (period === 'daily') return date.toISOString().split('T')[0];
  const day = date.getUTCDay() || 7;
  date.setUTCDate(date.getUTCDate() - day + 1);
  return date.toISOString().split('T')[0];
}

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

const dayLabel = (key) => new Date(`${key}T12:00:00Z`).toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' });
const runLabel = (run, attempt = '') => `${dayLabel(periodKey(commitTimestampFor(run), 'daily'))}\n${commitShaFor(run).slice(0, 8)}${attempt ? ` · ${attempt}` : ''}`;

function runSummary(run, filters) {
  const tests = selected(run, filters); const completedTests = tests.filter(completed);
  return { total: tests.length, completed: completedTests.length, failed: tests.filter((t) => t.status === 'failed').length,
    timeout: tests.filter((t) => t.status === 'timeout').length,
    duration: tests.length > 0 && completedTests.length === tests.length ? round(sum(completedTests)) : null,
    completionPercent: tests.length ? completedTests.length / tests.length * 100 : null };
}

function labeledRuns(data) {
  const runs = sortRunsByCommit(data.runs); const counts = new Map(); const seen = new Map();
  runs.forEach((run) => counts.set(commitShaFor(run), (counts.get(commitShaFor(run)) ?? 0) + 1));
  const labels = runs.map((run) => {
    const sha = commitShaFor(run); const attempt = (seen.get(sha) ?? 0) + 1; seen.set(sha, attempt);
    return runLabel(run, counts.get(sha) > 1 ? `${attempt}/${counts.get(sha)}` : '');
  });
  return { runs, labels };
}

// Canonical attempts are already scoped by the loader; sorting needs no test results.
export function selectRecentRunAttempts(data) {
  return [...data.runs].sort(compareRunExecution).reverse();
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
    aggregateDelta: percentage(candidateDuration, baselineDuration), counts };
}

export function selectBenchmarkSeries(data, filters, logicalTestId) {
  const { runs, labels } = labeledRuns(data);
  return { runs, labels, series: pairsFor(filters).map((pair, index) => {
    const records = runs.map((run) => {
      const test = (run?.tests ?? []).find((t) => testMatches(t, filters) && t.target === pair.target && t.mode === pair.mode && t.logicalTestId === logicalTestId);
      return test ? { run, test } : null;
    });
    return { ...pair, name: `${pair.target} ${pair.mode}`, color: targetColor(pair.target, index), records,
      points: records.map((record) => completed(record?.test) ? { value: record.test.durationSeconds, record } : null) };
  }) };
}

function scopedBenchmarkDefinitions(data, filters, resultPredicate) {
  const suites = new Map(data.testCatalog.map(({ id, suite }) => [id, suite]));
  // Resolve omitted result suites from the catalog for direct explorer callers.
  const ids = new Set(data.runs.flatMap((run) => (run.tests ?? [])
    .filter((test) => (!resultPredicate || resultPredicate(test))
      && testMatches({ ...test, suite: test.suite ?? suites.get(test.logicalTestId) }, filters))
    .map(({ logicalTestId }) => logicalTestId)));
  return data.testCatalog.filter(({ id }) => ids.has(id));
}

export function selectDefaultBenchmarks(data, filters) {
  return scopedBenchmarkDefinitions({ ...data, runs: data.runs.filter(hasCommitTimestamp) }, filters, completed).slice(0, 3);
}

export function selectBenchmarkCatalog(data, filters) {
  const available = scopedBenchmarkDefinitions(data, filters);
  return { all: data.testCatalog, available, hiddenCount: data.testCatalog.length - available.length };
}
