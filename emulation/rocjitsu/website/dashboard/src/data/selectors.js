import { targetColor } from '../utils/chartColors.js';
import { resolveHistoryRange } from '../config/historyRanges.js';
import { commitShaFor, commitTimestampFor, compareRunExecution, sortRunsByCommit } from './runOrdering.js';


const modesFor = (filters) => filters.modes ?? ['ST', 'MT'];
const completed = (test) => test?.status === 'completed' && Number.isFinite(test.durationSeconds);
const sum = (tests) => tests.reduce((total, test) => total + test.durationSeconds, 0);
const round = (value) => Number(value.toPrecision(15));
const percentage = (candidate, baseline) => Number.isFinite(candidate) && Number.isFinite(baseline) && baseline > 0 ? (candidate - baseline) / baseline * 100 : null;
const resultMap = (run) => new Map((run?.tests ?? []).map((test) => [test.testId, test]));
const testMatches = (test, filters) => filters.targets.includes(test.target) && filters.suites.includes(test.suite) && modesFor(filters).includes(test.mode);
const selected = (run, filters) => (run?.tests ?? []).filter((test) => testMatches(test, filters));
const pairsFor = (filters) => filters.targets.flatMap((target) => modesFor(filters).map((mode) => ({ target, mode, key: `${target}:${mode}` })));
const hasConfiguration = (run, target, mode) => (run?.configurations ?? []).some((c) => c.target === target && c.threadingMode === mode);

export function periodKey(timestamp, period = 'weekly') {
  const date = new Date(timestamp);
  if (period === 'monthly') return date.toISOString().slice(0, 7);
  if (period === 'daily') return date.toISOString().slice(0, 10);
  const day = date.getUTCDay() || 7;
  date.setUTCDate(date.getUTCDate() - day + 1);
  return date.toISOString().slice(0, 10);
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

function shiftDay(key, offset) {
  const date = new Date(`${key}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() + offset);
  return date.toISOString().slice(0, 10);
}
function dayKeys(start, end) {
  const keys = [];
  for (let key = start; key <= end; key = shiftDay(key, 1)) keys.push(key);
  return keys;
}
const dayLabel = (key) => new Date(`${key}T12:00:00Z`).toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' });
const runLabel = (run, attempt = '') => `${dayLabel(periodKey(commitTimestampFor(run), 'daily'))}\n${commitShaFor(run).slice(0, 8)}${attempt ? ` · ${attempt}` : ''}`;

function historySlots(runs, anchorDay, range) {
  const ordered = sortRunsByCommit(runs);
  if (!ordered.length) return { slots: [], keys: [], axisMax: 0 };
  if (range === '1D') {
    const commits = new Map();
    ordered.filter((r) => periodKey(commitTimestampFor(r), 'daily') === anchorDay).forEach((r) => commits.set(commitShaFor(r), r));
    const slots = [...commits.values()].slice(-56).map((run, x) => ({ run, dayKey: anchorDay, x, label: runLabel(run) }));
    return { slots, keys: null, axisMax: Math.max(0, slots.length - 1) };
  }
  const start = range === 'ALL' ? periodKey(commitTimestampFor(ordered[0]), 'daily')
    : range === 'YTD' ? `${anchorDay.slice(0, 4)}-01-01` : shiftDay(anchorDay, -(resolveHistoryRange(range, { compatibility: true }).days - 1));
  const keys = dayKeys(start, anchorDay);
  if (range === '1W') {
    const slots = keys.flatMap((key, index) => {
      const commits = new Map();
      ordered.filter((r) => periodKey(commitTimestampFor(r), 'daily') === key).forEach((r) => commits.set(commitShaFor(r), r));
      const entries = [...commits.values()].slice(-8);
      if (!entries.length) return [{ run: null, dayKey: key, x: index, label: dayLabel(key) }];
      return entries.map((run) => {
        const date = new Date(commitTimestampFor(run));
        const fraction = (date.getUTCHours() * 3600 + date.getUTCMinutes() * 60 + date.getUTCSeconds()) / 86400;
        return { run, dayKey: key, x: index + fraction, label: runLabel(run) };
      });
    });
    return { slots, keys, axisMax: keys.length };
  }
  const latest = new Map();
  ordered.forEach((run) => latest.set(periodKey(commitTimestampFor(run), 'daily'), run));
  return { slots: keys.map((key, x) => ({ run: latest.get(key) ?? null, dayKey: key, x,
    label: latest.has(key) ? runLabel(latest.get(key)) : dayLabel(key) })), keys, axisMax: Math.max(0, keys.length - 1) };
}

// The latest-commit catalog defines scope. Each target/mode has its own immutable,
// first-success anchor; only workloads absent from an older catalog may be estimated.
function currentWorkload(data, filters) {
  const definitions = new Map(data.testCatalog.map((test) => [test.id, test]));
  const ordered = [...data.runs].sort(compareRunExecution);
  const commitOrdered = sortRunsByCommit(data.runs);
  return pairsFor(filters).map((pair) => {
    const reference = commitOrdered.findLast((run) => Object.hasOwn(data.catalogs?.[run.testCatalog]?.configurations ?? {}, pair.key));
    const catalog = data.catalogs?.[reference?.testCatalog];
    const ids = (catalog?.configurations?.[pair.key] ?? [])
      .filter((id) => filters.suites.includes(definitions.get(id)?.suite));
    const selectedIds = new Set(ids);
    const anchors = new Map();
    for (const run of ordered) {
      if (run.catalogId !== reference?.catalogId) continue;
      for (const test of run.tests) if (test.target === pair.target && test.mode === pair.mode && selectedIds.has(test.logicalTestId) && completed(test) && !anchors.has(test.logicalTestId)) {
        anchors.set(test.logicalTestId, { testId: test.testId, logicalTestId: test.logicalTestId, target: pair.target, mode: pair.mode,
          durationSeconds: test.durationSeconds, runId: run.runId, catalogId: run.catalogId, timestamp: run.timestamp });
      }
    }
    // One eligible set is shared by every point, including the current endpoint.
    // A workload cannot contribute until its configuration has a valid anchor.
    return { ...pair, catalogId: reference?.catalogId, ids: ids.filter((id) => anchors.has(id)), anchors };
  }).filter(({ ids }) => ids.length > 0);
}

function adjustedDuration(data, run, workload) {
  if (!run || workload.length === 0) return { value: null, estimated: false, estimates: [] };
  const tests = resultMap(run); let total = 0; const estimates = [];
  for (const configuration of workload) {
    if (!hasConfiguration(run, configuration.target, configuration.mode)) return { value: null, estimated: false, estimates: [] };
    for (const id of configuration.ids) {
      const test = tests.get(`${configuration.key}:${id}`);
      if (test) {
        if (!completed(test)) return { value: null, estimated: false, estimates: [] };
        total += test.durationSeconds;
      } else {
        const anchor = configuration.anchors.get(id);
        const oldMembers = data.catalogs?.[run.testCatalog]?.configurations?.[configuration.key];
        if (run.catalogId === configuration.catalogId || oldMembers?.includes(id) || !anchor) return { value: null, estimated: false, estimates: [] };
        total += anchor.durationSeconds; estimates.push(anchor);
      }
    }
  }
  return { value: round(total), estimated: estimates.length > 0, estimates };
}

function runSummary(run, filters) {
  const tests = selected(run, filters); const completedTests = tests.filter(completed);
  return { total: tests.length, completed: completedTests.length, failed: tests.filter((t) => t.status === 'failed').length,
    timeout: tests.filter((t) => t.status === 'timeout').length,
    duration: tests.length > 0 && completedTests.length === tests.length ? round(sum(completedTests)) : null,
    completionPercent: tests.length ? completedTests.length / tests.length * 100 : null };
}

export function selectOverview(data, filters, range = 'ALL') {
  range = resolveHistoryRange(range, { compatibility: true }).id;
  const reference = data.latestCommitRun ?? sortRunsByCommit(data.runs).at(-1) ?? null;
  const anchorDay = periodKey(commitTimestampFor(reference) ?? data.generatedAt, 'daily');
  const { slots, keys, axisMax } = historySlots(data.runs, anchorDay, range);
  const workload = currentWorkload(data, filters);
  const projections = slots.map(({ run }) => adjustedDuration(data, run, workload));
  const firstIndex = projections.findIndex(({ value }) => Number.isFinite(value));
  const lastIndex = slots.findLastIndex(({ run }) => run);
  const firstRun = slots[firstIndex]?.run ?? null;
  const candidate = slots[lastIndex]?.run ?? null;
  const baselineDuration = projections[firstIndex]?.value ?? null;
  const currentDuration = projections[lastIndex]?.value ?? null;
  const durationDelta = firstIndex >= 0 && lastIndex !== firstIndex ? percentage(currentDuration, baselineDuration) : null;
  const baseline = firstRun?.runId !== candidate?.runId ? firstRun : null;
  const comparisons = compareRuns(candidate, baseline, filters);
  const summary = runSummary(candidate, filters);
  const representedDays = new Set(slots.filter(({ run }) => run).map(({ dayKey }) => dayKey)).size;
  const requestedDays = keys?.length ?? 0;
  const history = { range, mode: range === '1D' ? 'intraday' : range === '1W' ? 'weekly-by-commit' : 'daily-by-commit', anchorDay,
    slots, dayKeys: keys, axisMax, currentDuration, firstRun, latestRun: candidate, durationDelta,
    summary: `${slots.filter(({ run }) => run).length} commits shown`, normalized: projections.some(({ estimated }) => estimated),
    insufficientData: !['ALL', '1D', '1W'].includes(range) && requestedDays > 0 && representedDays / requestedDays < 0.75,
    anchors: workload.flatMap(({ anchors }) => [...anchors.values()]),
    estimates: projections.flatMap((projection, index) => projection.estimates.map((anchor) => ({ ...anchor, estimatedRunId: slots[index].run.runId }))),
    series: [{ key: 'selected-runtime', target: 'Selected runtime', color: targetColor('gfx1250', 0),
      data: projections.map(({ value }) => value), baseline: baselineDuration, estimated: projections.map(({ estimated }) => estimated) }] };
  return { candidate, baseline, changes: comparisons.filter(({ delta }) => Number.isFinite(delta)).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 6), history,
    results: comparisons.map((item) => ({ ...item.candidateTest, baselineTest: item.baselineTest, comparable: item.comparable, delta: item.delta })),
    metrics: { duration: currentDuration, durationDelta, completed: summary.completed, total: summary.total, failed: summary.failed + summary.timeout,
      completeness: summary.completionPercent ?? 0, estimatedBaseline: projections[firstIndex]?.estimated ?? false }, metricsBaseline: firstRun };
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

export function selectRunComparison(candidate, baseline, filters, tolerance = 3) {
  const candidates = compareRuns(candidate, baseline, filters); const ids = new Set(candidates.map(({ candidateTest }) => candidateTest.testId));
  const baselineOnly = selected(baseline, filters).filter(({ testId }) => !ids.has(testId)).map((baselineTest) => ({ candidateTest: null, baselineTest, comparable: false, delta: null, deltaSeconds: null }));
  const comparisons = [...candidates, ...baselineOnly]; const comparable = comparisons.filter(({ comparable }) => comparable);
  const baselineDuration = comparable.length ? round(sum(comparable.map(({ baselineTest }) => baselineTest))) : null;
  const candidateDuration = comparable.length ? round(sum(comparable.map(({ candidateTest }) => candidateTest))) : null;
  const counts = { faster: 0, slower: 0, neutral: 0, unavailable: 0 };
  for (const { delta } of comparable) counts[delta === null ? 'unavailable' : delta > tolerance ? 'slower' : delta < -tolerance ? 'faster' : 'neutral'] += 1;
  return { comparable: comparable.sort((a, b) => (Number.isFinite(b.delta) ? Math.abs(b.delta) : -1) - (Number.isFinite(a.delta) ? Math.abs(a.delta) : -1)),
    notComparable: comparisons.filter(({ comparable }) => !comparable), baselineDuration, candidateDuration,
    aggregateDelta: percentage(candidateDuration, baselineDuration), counts };
}

export function selectBenchmarkSeries(data, filters, logicalTestId) {
  const { runs, labels } = labeledRuns(data);
  return { runs, labels, series: pairsFor(filters).map((pair, index) => {
    const records = runs.map((run) => {
      const test = selected(run, filters).find((t) => t.target === pair.target && t.mode === pair.mode && t.logicalTestId === logicalTestId);
      return test ? { run, test } : null;
    });
    return { ...pair, name: `${pair.target} ${pair.mode}`, color: targetColor(pair.target, index), records,
      points: records.map((record) => completed(record?.test) ? { value: record.test.durationSeconds, record } : null) };
  }) };
}

export function selectBenchmarkCatalog(data, filters) {
  const suites = new Map(data.testCatalog.map(({ id, suite }) => [id, suite]));
  // Direct explorer callers may omit result-level suite metadata. Resolve it
  // from the catalog here so availability still uses the shared scope policy.
  const ids = new Set(data.runs.flatMap((run) => (run.tests ?? [])
    .filter((test) => testMatches({ ...test, suite: test.suite ?? suites.get(test.logicalTestId) }, filters))
    .map(({ logicalTestId }) => logicalTestId)));
  const available = data.testCatalog.filter(({ id }) => ids.has(id));
  return { all: data.testCatalog, available, hiddenCount: data.testCatalog.length - available.length };
}
