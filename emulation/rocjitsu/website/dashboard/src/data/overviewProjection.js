import { resolveHistoryRange } from '../config/historyRanges.js';
import { commitShaFor, commitTimestampFor, sortRunsByCommit } from './runOrdering.js';
import { configurationKey, hasConfiguration, THREAD_POLICIES } from './runSchema.js';

const utcDay = (timestamp) => typeof timestamp === 'string' && Number.isFinite(Date.parse(timestamp)) ? new Date(timestamp).toISOString().split('T')[0] : null;
const dayTime = (day) => Date.parse(`${day}T00:00:00Z`);
const dayFor = (run) => utcDay(commitTimestampFor(run));
const shiftDay = (day, offset) => {
  const date = new Date(`${day}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() + offset);
  return date.toISOString().split('T')[0];
};

function dateWindow(runs, generatedAt, range) {
  const anchorDay = runs.length ? dayFor(runs.at(-1)) : utcDay(generatedAt);
  if (!anchorDay) return { anchorDay: null, startDay: null, dayKeys: [] };
  const startDay = range.id === 'ALL' ? (runs.length ? dayFor(runs[0]) : anchorDay)
    : range.id === 'YTD' ? `${anchorDay.slice(0, -6)}-01-01` : shiftDay(anchorDay, -(range.days - 1));
  const dayKeys = [];
  for (let day = startDay; dayTime(day) <= dayTime(anchorDay); day = shiftDay(day, 1)) {
    dayKeys.push(day);
    if (day === anchorDay) break;
  }
  return { anchorDay, startDay, dayKeys };
}

function selectRuns(runs, range) {
  if (range === '1W') return runs;
  const selected = new Map();
  for (const run of runs) selected.set(range === '1D' ? commitShaFor(run) : dayFor(run), run);
  const values = sortRunsByCommit([...selected.values()]);
  return range === '1D' ? values.slice(-56) : values;
}

function configurationRows(data, filters, runs) {
  const targets = [...new Set(filters.targets)].sort();
  return THREAD_POLICIES.filter((mode) => (filters.modes ?? THREAD_POLICIES).includes(mode))
    .flatMap((mode) => targets.map((target) => {
      const key = configurationKey(target, mode);
      const catalogs = new Map();
      for (const run of runs) {
        const catalog = data.catalogs?.[run.catalogId];
        if (catalog && hasConfiguration(run, target, mode) && Object.hasOwn(catalog.configurations, key)) {
          catalogs.set(catalog.id, catalog);
        }
      }
      const applicable = [...catalogs.values()];
      const first = applicable[0];
      const definitions = new Map((first?.tests ?? []).map((test) => [test.id, test]));
      const memberships = applicable.map((catalog) => new Set(catalog.configurations[key]));
      const cohortIds = (first?.configurations[key] ?? []).filter((id) => memberships.every((ids) => ids.has(id))
        && filters.suites.includes(definitions.get(id)?.suite)).sort();
      return { key, target, mode, thread_policy: mode, catalogIds: [...catalogs.keys()], cohortIds,
        comparedWorkloads: cohortIds.length };
    }));
}

function projectPoint(run, row) {
  const identity = { run, runId: run?.runId ?? null, commit: run ? commitShaFor(run) : null,
    commitTimestamp: run ? commitTimestampFor(run) : null, completedAt: run?.timestamp ?? null,
    dayKey: run ? dayFor(run) : null };
  const required = row.cohortIds.length;
  const point = { ...identity, required, completed: 0, durationSeconds: null, speedup: null, results: [], reasons: [] };
  if (!run) return { ...point, reason: 'no-run' };
  if (!required) return { ...point, reason: 'no-comparable-workloads' };
  if (!hasConfiguration(run, row.target, row.mode)) return { ...point, reason: 'configuration-absent' };
  const results = new Map(run.tests.filter((test) => test.target === row.target && test.mode === row.mode)
    .map((test) => [test.logicalTestId, test]));
  let total = 0;
  for (const id of row.cohortIds) {
    const result = results.get(id);
    const reason = !result ? 'missing-result' : result.status !== 'completed' ? result.status
      : !Number.isFinite(result.durationSeconds) || result.durationSeconds <= 0 ? 'invalid-duration' : null;
    point.results.push({ id, result: result ?? null, reason });
    if (reason) point.reasons.push({ id, reason });
    else { point.completed += 1; total += result.durationSeconds; }
  }
  if (point.reasons.length) return { ...point, reason: point.reasons[0].reason };
  if (!Number.isFinite(total) || total <= 0) return { ...point, reason: 'invalid-total' };
  return { ...point, reason: null, durationSeconds: Number(total.toPrecision(15)) };
}

function createSlots(dayKeys, runs, range) {
  if (range === '1D') return runs.map((run, x) => ({ run, x, dayKey: dayFor(run) }));
  const groups = new Map();
  for (const run of runs) {
    const day = dayFor(run);
    if (!groups.has(day)) groups.set(day, []);
    groups.get(day).push(run);
  }
  return dayKeys.flatMap((dayKey, x) => {
    const entries = groups.get(dayKey) ?? [];
    if (!entries.length) return [{ run: null, dayKey, x }];
    return entries.map((run, index) => ({ run, dayKey,
      x: range === '1W' ? x + (index + 1) / (entries.length + 1) : x }));
  });
}

// Shared Overview projection; catalog scoping preserves the UTC window.
// catalogId=null selects Shared benchmarks; an explicit immutable catalog ID scopes
// candidate runs without moving the UTC window. Hover is intentionally not input.
export function selectOverviewProjection(data, filters, { range = 'ALL', catalogId = null } = {}) {
  const resolved = resolveHistoryRange(range, { compatibility: true });
  const developRuns = sortRunsByCommit(data.runs.filter((run) => run.branch === (data.canonicalBranch ?? 'develop')));
  const window = dateWindow(developRuns, data.runs.length ? null : data.generatedAt, resolved);
  const withinWindow = developRuns.filter((run) => dayTime(dayFor(run)) >= dayTime(window.startDay) && dayTime(dayFor(run)) <= dayTime(window.anchorDay)
    && (catalogId === null || run.catalogId === catalogId));
  const candidates = selectRuns(withinWindow, resolved.id);
  const configurations = configurationRows(data, filters, candidates);
  // Membership is fixed before weekly success eligibility; failed catalogs remain
  // participants and each configuration has an independent range-wide cohort.
  const attempts = candidates.map((run) => {
    const points = configurations.map((row) => projectPoint(run, row));
    return { run, runId: run.runId, points, eligible: points.length > 0 && points.every((point) => point.durationSeconds !== null) };
  });
  const attemptPoints = new Map(attempts.map(({ run, points }) => [run, points]));
  const plotted = resolved.id === '1W' ? attempts.filter(({ eligible }) => eligible).map(({ run }) => run) : candidates;
  const slots = createSlots(window.dayKeys, plotted, resolved.id);
  const latestIndex = slots.findLastIndex(({ run }) => run !== null);
  const rows = configurations.map((row, rowIndex) => {
    const points = slots.map(({ run, dayKey }) => {
      const point = run ? attemptPoints.get(run)[rowIndex] : projectPoint(null, row);
      // Row speedups and mutable wrappers are independent; source leaves stay original.
      return { ...point, dayKey, results: point.results.map((result) => ({ ...result })),
        reasons: point.reasons.map((reason) => ({ ...reason })) };
    });
    const reference = points.find((point) => point.durationSeconds > 0) ?? null;
    for (const point of points) {
      const ratio = reference && point.durationSeconds > 0 ? reference.durationSeconds / point.durationSeconds : null;
      point.speedup = Number.isFinite(ratio) ? ratio : null;
    }
    const latest = points[latestIndex] ?? projectPoint(null, row);
    const change = reference && latest.durationSeconds > 0
      ? (latest.durationSeconds - reference.durationSeconds) / reference.durationSeconds * 100 : null;
    return { ...row, points, reference, latest,
      endpoints: { reference, latest }, durationChangePercent: Number.isFinite(change) ? change : null };
  });
  return { range: resolved.id, catalogId, ...window, slots, attempts, rows,
    latestRun: slots[latestIndex]?.run ?? null,
    endpointLabel: resolved.id === '1W' ? 'Latest eligible plotted run' : 'Latest selected run',
    weeklyEligibility: resolved.id === '1W' ? 'selected-workloads-succeed' : null,
    axisMax: resolved.id === '1W' ? window.dayKeys.length : Math.max(0, slots.length - 1),
    defaultConfigurationKey: rows[0]?.key ?? null };
}
