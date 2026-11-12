import { configurationKey, THREAD_POLICIES } from './runSchema.js';
import { completed, testMatches } from '../../../../scripts/benchmark-data/comparison.mjs';
export { compareRuns, selectRunComparison, selectRecentRunSummaries } from '../../../../scripts/benchmark-data/comparison.mjs';
import { commitShaFor, commitTimestampFor, compareRunExecution, hasCommitTimestamp, sortRunsByCommit } from './runOrdering.js';


const modesFor = (filters) => filters.modes ?? THREAD_POLICIES;
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

const dayLabel = (key) => new Date(`${key}T12:00:00Z`).toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' });
const runLabel = (run, attempt = '') => `${dayLabel(periodKey(commitTimestampFor(run), 'daily'))}\n${commitShaFor(run).slice(0, 8)}${attempt ? ` · ${attempt}` : ''}`;

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

export function selectBenchmarkSeries(data, filters, logicalTestId) {
  const { runs, labels } = labeledRuns(data);
  return { runs, labels, series: pairsFor(filters).map((pair) => {
    const records = runs.map((run) => {
      const test = (run?.tests ?? []).find((t) => testMatches(t, filters) && t.target === pair.target && t.mode === pair.mode && t.logicalTestId === logicalTestId);
      return test ? { run, test } : null;
    });
    return { ...pair, name: `${pair.target} ${pair.mode}`, records,
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
