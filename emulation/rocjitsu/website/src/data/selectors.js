import { targetColor } from '../utils/chartColors';
import {
  commitShaFor,
  commitTimestampFor,
  compareCommitPosition,
  compareRunsByCommit,
  sortRunsByCommit,
} from './runOrdering';

const HISTORY_RANGE_DAYS = {
  '1W': 7,
  '1M': 30,
  '3M': 90,
  '6M': 180,
};
const MAX_COMMITS_PER_DAY = 8;
const MAX_INTRADAY_COMMITS = 56;
const MIN_HISTORY_COVERAGE = 0.75;

export function periodKey(timestamp, period = 'weekly') {
  const date = new Date(timestamp);
  if (period === 'monthly') return date.toISOString().slice(0, 7);
  if (period === 'daily') return date.toISOString().slice(0, 10);
  const day = date.getUTCDay() || 7;
  date.setUTCDate(date.getUTCDate() - day + 1);
  return date.toISOString().slice(0, 10);
}

export function testMatches(test, filters) {
  return filters.targets.includes(test.target) && filters.suites.includes(test.suite);
}

export function isRunCompletedForFilters(run, filters) {
  const selectedTests = (run?.tests ?? []).filter((test) => testMatches(test, filters));
  return selectedTests.length > 0
    && selectedTests.every((test) => (
      test.status === 'completed' && Number.isFinite(test.durationSeconds)
    ));
}

export function resultMap(run) {
  return new Map((run?.tests ?? []).map((test) => [test.testId, test]));
}

export function previousCompletedRunForFilters(runs, candidate, filters) {
  if (!candidate) return null;
  const selectedTestIds = candidate.tests
    .filter((test) => testMatches(test, filters))
    .map((test) => test.testId);
  if (selectedTestIds.length === 0) return null;

  return runs.filter((run) => {
    if (compareCommitPosition(run, candidate) >= 0) return false;
    if (!isRunCompletedForFilters(run, filters)) return false;
    const tests = resultMap(run);
    return selectedTestIds.every((testId) => {
      const test = tests.get(testId);
      return test != null;
    });
  }).sort(compareRunsByCommit).at(-1) ?? null;
}

export function previousCompletedTestResult(runs, candidate, testId) {
  if (!candidate || !testId) return null;
  const earlierRuns = runs.filter((run) => (
    compareCommitPosition(run, candidate) < 0
  )).sort(compareRunsByCommit).reverse();

  for (const run of earlierRuns) {
    const test = run.tests.find((candidateTest) => candidateTest.testId === testId);
    if (test?.status === 'completed' && Number.isFinite(test.durationSeconds)) {
      return { run, test };
    }
  }
  return null;
}

export function compareRuns(candidate, baseline, filters) {
  if (!candidate) return [];
  const baselineTests = resultMap(baseline);
  return candidate.tests.filter((test) => testMatches(test, filters)).map((test) => {
    const previous = baselineTests.get(test.testId);
    let notComparableReason = null;
    if (!previous) notComparableReason = 'Missing from baseline';
    else if (test.status !== 'completed') notComparableReason = `Candidate ${test.status ?? 'incomplete'}`;
    else if (previous.status !== 'completed') notComparableReason = `Baseline ${previous.status ?? 'incomplete'}`;
    else if (!Number.isFinite(test.durationSeconds)) notComparableReason = 'Candidate duration unavailable';
    else if (!Number.isFinite(previous.durationSeconds)) notComparableReason = 'Baseline duration unavailable';

    const comparable = notComparableReason === null;
    return {
      test,
      previous,
      comparable,
      notComparableReason,
      delta: comparable ? ((test.durationSeconds - previous.durationSeconds) / previous.durationSeconds) * 100 : null,
    };
  });
}

function sumDurations(tests) {
  return tests.reduce((total, test) => total + (test.status === 'completed' && Number.isFinite(test.durationSeconds) ? test.durationSeconds : 0), 0);
}

function shiftUtcDay(dayKey, offset) {
  const date = new Date(`${dayKey}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() + offset);
  return date.toISOString().slice(0, 10);
}

function calendarDayKeys(startKey, endKey) {
  const days = [];
  for (let key = startKey; key <= endKey; key = shiftUtcDay(key, 1)) days.push(key);
  return days;
}

function historyStartKey(runs, anchorKey, range, timestampForRun) {
  if (range === 'ALL') return periodKey(timestampForRun(runs[0]) ?? `${anchorKey}T00:00:00Z`, 'daily');
  if (range === 'YTD') return `${anchorKey.slice(0, 4)}-01-01`;
  const dayCount = HISTORY_RANGE_DAYS[range] ?? HISTORY_RANGE_DAYS['3M'];
  return shiftUtcDay(anchorKey, -(dayCount - 1));
}

function shortRunSha(run) {
  return commitShaFor(run).slice(0, 8) || 'unknown';
}

function shortDayLabel(dayKey) {
  return new Date(`${dayKey}T12:00:00Z`).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    timeZone: 'UTC',
  });
}

function benchmarkRunLabel(run, attemptLabel = '') {
  const dayKey = periodKey(commitTimestampFor(run), 'daily');
  return `${shortDayLabel(dayKey)}\n${shortRunSha(run)}${attemptLabel ? ` · ${attemptLabel}` : ''}`;
}

function durationForRun(run, target, suites) {
  if (!run) return null;
  const selectedTests = run.tests.filter((test) => test.target === target && suites.includes(test.suite));
  if (
    selectedTests.length === 0
    || selectedTests.some((test) => test.status !== 'completed' || !Number.isFinite(test.durationSeconds))
  ) return null;
  return Number(sumDurations(selectedTests).toFixed(3));
}

function selectedTestsForTarget(run, target, suites) {
  return (run?.tests ?? []).filter((test) => (
    test.target === target && suites.includes(test.suite)
  ));
}

function historyWorkload(run, target, suites) {
  return new Set(selectedTestsForTarget(run, target, suites).map((test) => test.logicalTestId));
}

function latestCompletedResultAnchors(runs, target, suites, logicalTestIds) {
  const anchors = new Map();
  for (const run of sortRunsByCommit(runs).reverse()) {
    for (const test of selectedTestsForTarget(run, target, suites)) {
      if (
        logicalTestIds.has(test.logicalTestId)
        && test.status === 'completed'
        && Number.isFinite(test.durationSeconds)
        && !anchors.has(test.logicalTestId)
      ) {
        anchors.set(test.logicalTestId, test.durationSeconds);
      }
    }
  }
  return anchors;
}

function normalizedDurationForRun(
  run,
  target,
  suites,
  canonicalTestIds,
  canonicalCatalogId,
  anchors,
) {
  if (!run || canonicalTestIds.size === 0) return { value: null, estimatedTests: [] };
  const tests = new Map(selectedTestsForTarget(run, target, suites)
    .map((test) => [test.logicalTestId, test]));
  const estimatedTests = [];
  let total = 0;

  for (const logicalTestId of canonicalTestIds) {
    const test = tests.get(logicalTestId);
    if (test) {
      if (test.status !== 'completed' || !Number.isFinite(test.durationSeconds)) {
        return { value: null, estimatedTests: [] };
      }
      total += test.durationSeconds;
      continue;
    }

    const anchor = run.catalogId !== canonicalCatalogId ? anchors.get(logicalTestId) : null;
    if (!Number.isFinite(anchor)) return { value: null, estimatedTests: [] };
    total += anchor;
    estimatedTests.push(logicalTestId);
  }

  return {
    value: Number(total.toFixed(3)),
    estimatedTests,
  };
}

function latestCompletedRunForCommit(runs, referenceRun) {
  if (!referenceRun) return null;
  const commit = commitShaFor(referenceRun);
  return sortRunsByCommit(runs.filter((run) => commitShaFor(run) === commit)).at(-1) ?? null;
}

function dailyHistorySlots(runs, anchorDay, range) {
  const commitOrderedRuns = sortRunsByCommit(runs);
  const startKey = historyStartKey(commitOrderedRuns, anchorDay, range, commitTimestampFor);
  const latestRunByDay = new Map();
  commitOrderedRuns.forEach((run) => {
    const key = periodKey(commitTimestampFor(run), 'daily');
    if (key >= startKey && key <= anchorDay) latestRunByDay.set(key, run);
  });
  return calendarDayKeys(startKey, anchorDay).map((dayKey) => {
    const run = latestRunByDay.get(dayKey) ?? null;
    return {
      dayKey,
      run,
      label: run ? `${shortDayLabel(dayKey)}\n${shortRunSha(run)}` : shortDayLabel(dayKey),
    };
  });
}

function intradayHistorySlots(runs, anchorDay) {
  const runsByCommit = new Map();
  runs
    .filter((run) => periodKey(run.timestamp, 'daily') === anchorDay)
    .sort(compareRunsByCommit)
    .forEach((run) => runsByCommit.set(commitShaFor(run), run));
  return [...runsByCommit.values()]
    .sort(compareRunsByCommit)
    .slice(-MAX_INTRADAY_COMMITS)
    .map((run) => ({
      dayKey: anchorDay,
      run,
      label: benchmarkRunLabel(run),
    }));
}

function weeklyHistorySlots(runs, anchorDay) {
  const startKey = shiftUtcDay(anchorDay, -(HISTORY_RANGE_DAYS['1W'] - 1));
  return calendarDayKeys(startKey, anchorDay).flatMap((dayKey) => {
    const runsByCommit = new Map();
    runs
      .filter((run) => periodKey(commitTimestampFor(run), 'daily') === dayKey)
      .sort(compareRunsByCommit)
      .forEach((run) => runsByCommit.set(commitShaFor(run), run));
    const dailyRuns = [...runsByCommit.values()]
      .sort(compareRunsByCommit)
      .slice(-MAX_COMMITS_PER_DAY)
    return Array.from({ length: MAX_COMMITS_PER_DAY }, (_, index) => ({
      dayKey,
      run: dailyRuns[index] ?? null,
      label: index === 0 ? shortDayLabel(dayKey) : '',
    }));
  });
}

export function selectOverview(data, filters, range = 'ALL') {
  const completedRuns = data.runs.filter((run) => isRunCompletedForFilters(run, filters));
  const candidate = data.latestCommitRun ?? sortRunsByCommit(data.runs).at(-1) ?? data.latestRun;
  const trendRuns = completedRuns;
  const baseline = previousCompletedRunForFilters(data.runs, candidate, filters);
  const comparisons = compareRuns(candidate, baseline, filters);
  const latestTests = (candidate?.tests ?? []).filter((test) => testMatches(test, filters));
  const completedTests = latestTests.filter((test) => test.status === 'completed');
  const totalTestCount = latestTests.length;
  const candidateComplete = totalTestCount > 0
    && completedTests.length === totalTestCount
    && completedTests.every((test) => Number.isFinite(test.durationSeconds));
  const failed = latestTests.filter((test) => test.status !== 'completed').length;

  const isIntraday = range === '1D';
  const isWeekly = range === '1W';
  const anchorDay = periodKey(isIntraday ? candidate.timestamp : commitTimestampFor(candidate), 'daily');
  const intradayRuns = completedRuns;
  const slots = isIntraday
    ? intradayHistorySlots(intradayRuns, anchorDay)
    : isWeekly
      ? weeklyHistorySlots(completedRuns, anchorDay)
      : dailyHistorySlots(trendRuns, anchorDay, range);
  const representedRuns = slots.filter((slot) => slot.run).length;
  const representedDays = new Set(slots.filter((slot) => slot.run).map((slot) => slot.dayKey)).size;
  const requestedDays = range === 'YTD'
    ? Math.floor((
      Date.parse(`${anchorDay}T00:00:00Z`)
      - Date.parse(`${anchorDay.slice(0, 4)}-01-01T00:00:00Z`)
    ) / 86_400_000) + 1
    : HISTORY_RANGE_DAYS[range];
  const insufficientData = !isIntraday
    && !isWeekly
    && range !== 'ALL'
    && requestedDays
    && representedDays / requestedDays < MIN_HISTORY_COVERAGE;
  const historyCandidate = [...slots].reverse().find((slot) => slot.run)?.run ?? candidate;
  const firstHistoryRun = slots.find((slot) => slot.run)?.run ?? null;
  const previousIntradayRun = isIntraday && firstHistoryRun
    ? sortRunsByCommit(completedRuns.filter((run) => (
      periodKey(commitTimestampFor(run), 'daily') === shiftUtcDay(anchorDay, -1)
    ))).at(-1) ?? null
    : null;
  const candidateHistoryBaseline = isIntraday
    ? previousIntradayRun ?? firstHistoryRun
    : firstHistoryRun;
  const historyBaseline = !insufficientData
    && candidateHistoryBaseline?.runId !== historyCandidate?.runId
    ? candidateHistoryBaseline
    : null;
  const displayedDuration = sumDurations(historyCandidate.tests.filter((test) => testMatches(test, filters)));
  const normalizedSeries = filters.targets.map((target, index) => {
    const canonicalTestIds = historyWorkload(candidate, target, filters.suites);
    const anchors = latestCompletedResultAnchors(data.runs, target, filters.suites, canonicalTestIds);
    const projected = slots.map((slot) => normalizedDurationForRun(
      slot.run,
      target,
      filters.suites,
      canonicalTestIds,
      candidate?.catalogId,
      anchors,
    ));
    const firstValue = projected.find(({ value }) => Number.isFinite(value))?.value ?? null;
    const previousValue = previousIntradayRun
      ? normalizedDurationForRun(
        previousIntradayRun,
        target,
        filters.suites,
        canonicalTestIds,
        candidate?.catalogId,
        anchors,
      ).value
      : null;
    return {
      target,
      color: targetColor(target, index),
      data: projected.map(({ value }) => value),
      baseline: isIntraday && Number.isFinite(previousValue) ? previousValue : firstValue,
      estimated: projected.map(({ estimatedTests }) => estimatedTests.length > 0),
      estimatedTests: projected.map(({ estimatedTests }) => estimatedTests),
    };
  });
  const normalizedDurationAt = (index) => {
    const values = normalizedSeries.map((series) => series.data[index]);
    return values.every(Number.isFinite)
      ? values.reduce((total, value) => total + value, 0)
      : null;
  };
  const historyCandidateIndex = slots.findLastIndex((slot) => slot.run?.runId === historyCandidate?.runId);
  const firstHistoryIndex = slots.findIndex((slot) => slot.run?.runId === firstHistoryRun?.runId);
  const normalizedCandidateDuration = normalizedDurationAt(historyCandidateIndex);
  const normalizedBaselineDuration = isIntraday
    ? normalizedSeries.every((series) => Number.isFinite(series.baseline))
      ? normalizedSeries.reduce((total, series) => total + series.baseline, 0)
      : null
    : normalizedDurationAt(firstHistoryIndex);
  const oldestCompletedRun = sortRunsByCommit(completedRuns)[0] ?? null;
  const metricsBaseline = latestCompletedRunForCommit(completedRuns, oldestCompletedRun);
  const metricsBaselineDuration = metricsBaseline
    ? filters.targets.reduce((total, target) => {
      if (total == null) return null;
      const canonicalTestIds = historyWorkload(candidate, target, filters.suites);
      const anchors = latestCompletedResultAnchors(data.runs, target, filters.suites, canonicalTestIds);
      const normalized = normalizedDurationForRun(
        metricsBaseline,
        target,
        filters.suites,
        canonicalTestIds,
        candidate?.catalogId,
        anchors,
      );
      return normalized.value == null ? null : total + normalized.value;
    }, 0)
    : null;
  const metricsDurationDelta = candidateComplete
    && Number.isFinite(metricsBaselineDuration)
    && metricsBaselineDuration > 0
    ? ((sumDurations(completedTests) - metricsBaselineDuration) / metricsBaselineDuration) * 100
    : null;
  const historyComparisons = historyBaseline
    ? compareRuns(historyCandidate, historyBaseline, filters).filter((item) => item.comparable)
    : [];
  const history = {
    range,
    mode: isIntraday ? 'intraday' : isWeekly ? 'weekly-by-commit' : 'daily-by-commit',
    anchorDay,
    slots,
    runCount: representedRuns,
    currentDuration: displayedDuration,
    firstRun: historyBaseline ?? firstHistoryRun,
    latestRun: historyCandidate,
    durationDelta: !insufficientData
      && normalizedCandidateDuration != null
      && normalizedBaselineDuration
      ? ((normalizedCandidateDuration - normalizedBaselineDuration) / normalizedBaselineDuration) * 100
      : null,
    comparisonLabel: historyBaseline
      ? `Latest vs first shown in ${range}${normalizedSeries.some((series) => series.estimated.some(Boolean)) ? ' · normalized workload' : ''}`
      : 'At least two completed runs are needed',
    summary: `${representedRuns} commit${representedRuns === 1 ? '' : 's'} shown`,
    description: [
      isIntraday
        ? `All official commits from ${shortDayLabel(anchorDay)} (UTC) are shown`
        : isWeekly
          ? `All official commits are shown, up to ${MAX_COMMITS_PER_DAY} per UTC date`
          : 'The latest completed official run for each UTC commit date is shown',
      normalizedSeries.some((series) => series.estimated.some(Boolean))
        ? 'Historical values are normalized to the latest selected workload'
        : null,
    ].filter(Boolean).join('. '),
    normalized: normalizedSeries.some((series) => series.estimated.some(Boolean)),
    insufficientData,
    series: normalizedSeries,
  };

  const changes = historyComparisons
    .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta))
    .slice(0, 6);

  return {
    candidate,
    baseline,
    latestTests,
    comparisons,
    changes,
    history,
    results: comparisons.map((item) => ({ ...item, ...item.test })),
    metrics: {
      duration: candidateComplete ? sumDurations(completedTests) : null,
      durationDelta: metricsDurationDelta,
      completed: completedTests.length,
      total: totalTestCount,
      failed,
      completeness: totalTestCount ? (completedTests.length / totalTestCount) * 100 : 0,
    },
    metricsBaseline,
  };
}

export function selectAggregateRunSeries(data, filters) {
  const runs = sortRunsByCommit(data.runs);
  const attemptsPerCommit = runs.reduce((counts, run) => {
    const sha = commitShaFor(run);
    counts.set(sha, (counts.get(sha) ?? 0) + 1);
    return counts;
  }, new Map());
  const seenAttempts = new Map();

  return {
    runs,
    series: filters.targets.map((target, index) => ({
      target,
      color: targetColor(target, index),
      data: runs.map((run) => {
        const selectedTests = selectedTestsForTarget(run, target, filters.suites);
        return {
          value: durationForRun(run, target, filters.suites),
          run,
          target,
          completed: selectedTests.filter((test) => test.status === 'completed').length,
          total: selectedTests.length,
        };
      }),
      catalogBreaks: runs.reduce((breaks, run, index) => {
        if (index > 0 && run.catalogId !== runs[index - 1].catalogId) breaks.push(index);
        return breaks;
      }, []),
    })),
    labels: runs.map((run) => {
      const sha = commitShaFor(run);
      const attempt = (seenAttempts.get(sha) ?? 0) + 1;
      seenAttempts.set(sha, attempt);
      const total = attemptsPerCommit.get(sha);
      return benchmarkRunLabel(run, total > 1 ? `${attempt}/${total}` : '');
    }),
  };
}

function runSummary(run, filters) {
  const tests = (run?.tests ?? []).filter((test) => testMatches(test, filters));
  const completed = tests.filter((test) => test.status === 'completed');
  const failed = tests.filter((test) => test.status === 'failed').length;
  const timeout = tests.filter((test) => test.status === 'timeout').length;
  const total = tests.length;
  return {
    total,
    completed: completed.length,
    failed,
    timeout,
    duration: total > 0 && completed.length === total ? sumDurations(completed) : null,
    completionPercent: total ? (completed.length / total) * 100 : null,
  };
}

export function selectRecentRuns(data, filters, limit = 8) {
  return data.runs.slice(-limit).reverse().map((run, index) => {
    const summary = runSummary(run, filters);
    const baseline = previousCompletedRunForFilters(data.runs, run, filters);
    const comparable = compareRuns(run, baseline, filters).filter((item) => item.comparable);
    const candidateDuration = comparable.reduce((total, item) => total + item.test.durationSeconds, 0);
    const baselineDuration = comparable.reduce((total, item) => total + item.previous.durationSeconds, 0);
    return {
      run,
      baseline,
      ...summary,
      latest: index === 0,
      latestCommit: compareCommitPosition(run, data.latestCommitRun) === 0,
      olderCommit: data.backfillRunIds.has(run.runId),
      durationDelta: summary.completed === summary.total && baselineDuration
        ? ((candidateDuration - baselineDuration) / baselineDuration) * 100
        : null,
    };
  });
}

export function selectRunReliability(data, filters, limit = 20) {
  const runs = data.runs.slice(-limit);
  const rows = runs.map((run) => ({ run, ...runSummary(run, filters) }));
  const totals = rows.reduce((summary, row) => ({
    completed: summary.completed + row.completed,
    total: summary.total + row.total,
    failed: summary.failed + row.failed,
    timeout: summary.timeout + row.timeout,
  }), { completed: 0, total: 0, failed: 0, timeout: 0 });
  const fullyCompleteRuns = rows.filter((row) => row.total > 0 && row.completed === row.total).length;
  return {
    rows,
    issueRuns: rows.filter((row) => row.completed < row.total).reverse(),
    runCount: rows.length,
    fullyCompleteRuns,
    completionPercent: totals.total ? (totals.completed / totals.total) * 100 : null,
    failed: totals.failed,
    timeout: totals.timeout,
  };
}

export function selectRunComparison(candidate, baseline, filters, tolerance = 3) {
  const comparisons = compareRuns(candidate, baseline, filters);
  const comparable = comparisons.filter((item) => item.comparable);
  const notComparable = comparisons.filter((item) => !item.comparable);
  const baselineDuration = comparable.reduce((total, item) => total + item.previous.durationSeconds, 0);
  const candidateDuration = comparable.reduce((total, item) => total + item.test.durationSeconds, 0);
  const counts = comparable.reduce((summary, item) => {
    const state = item.delta > tolerance ? 'slower' : item.delta < -tolerance ? 'faster' : 'neutral';
    return { ...summary, [state]: summary[state] + 1 };
  }, { faster: 0, slower: 0, neutral: 0 });
  return {
    comparisons,
    comparable: [...comparable].sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta)),
    notComparable,
    baselineDuration,
    candidateDuration,
    aggregateDelta: baselineDuration
      ? ((candidateDuration - baselineDuration) / baselineDuration) * 100
      : null,
    counts,
  };
}

export function selectBenchmarkSeries(data, filters, logicalTestId) {
  const runs = sortRunsByCommit(data.runs);
  const attemptsPerCommit = runs.reduce((counts, run) => {
    const sha = commitShaFor(run);
    counts.set(sha, (counts.get(sha) ?? 0) + 1);
    return counts;
  }, new Map());
  const seenAttempts = new Map();
  return {
    runs,
    labels: runs.map((run) => {
      const sha = commitShaFor(run);
      const attempt = (seenAttempts.get(sha) ?? 0) + 1;
      seenAttempts.set(sha, attempt);
      const total = attemptsPerCommit.get(sha);
      return benchmarkRunLabel(run, total > 1 ? `${attempt}/${total}` : '');
    }),
    series: filters.targets.map((target, index) => {
      const records = runs.map((run) => {
        const test = run.tests.find((candidate) => candidate.target === target && candidate.logicalTestId === logicalTestId);
        return test ? { run, test } : null;
      });
      return {
        name: target,
        color: targetColor(target, index),
        records,
        points: records.map((record) => (
          record?.test.status === 'completed' && Number.isFinite(record.test.durationSeconds) ? {
            value: record.test.durationSeconds,
            record,
          } : null
        )),
      };
    }),
  };
}

export function selectBenchmarkRecords(data, filters, logicalTestId, offset = 0, limit = 25) {
  const records = [];
  let total = 0;
  const pageEnd = offset + limit;

  for (let runIndex = data.runs.length - 1; runIndex >= 0; runIndex -= 1) {
    const run = data.runs[runIndex];
    for (const test of run.tests) {
      if (!filters.targets.includes(test.target) || test.logicalTestId !== logicalTestId) continue;
      if (total >= offset && total < pageEnd) {
        const baselineResult = previousCompletedTestResult(data.runs, run, test.testId);
        const baseline = baselineResult?.run ?? null;
        const previous = baselineResult?.test ?? null;
        const comparable = test.status === 'completed'
          && previous?.status === 'completed'
          && Number.isFinite(test.durationSeconds)
          && Number.isFinite(previous.durationSeconds)
          && previous.durationSeconds !== 0;
        records.push({
          run,
          test,
          baseline,
          delta: comparable
            ? ((test.durationSeconds - previous.durationSeconds) / previous.durationSeconds) * 100
            : null,
        });
      }
      total += 1;
    }
  }

  return { records, total };
}

export function selectBenchmarkCatalog(data, filters) {
  const available = data.testCatalog.filter((test) => filters.suites.includes(test.suite));
  return {
    all: data.testCatalog,
    available,
    hiddenCount: data.testCatalog.length - available.length,
  };
}

export function selectFailures(data, filters) {
  return data.runs.flatMap((run) => run.tests
    .filter((test) => test.status !== 'completed' && testMatches(test, filters))
    .map((test) => ({ run, test }))).reverse();
}
