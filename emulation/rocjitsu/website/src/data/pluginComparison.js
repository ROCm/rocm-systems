import { compareRunsByCommit } from './runOrdering';

export const PLUGIN_NOISE_TOLERANCE = 3;

function completed(test) {
  return test?.status === 'completed' && Number.isFinite(test.durationSeconds);
}

function selectedTests(run, target, suites) {
  return (run?.tests ?? []).filter((test) => test.target === target && suites.includes(test.suite));
}

function durationTotal(tests) {
  return tests.reduce((total, test) => total + test.durationSeconds, 0);
}

export function selectPluginComparisonGroups(data) {
  const groups = new Map();
  for (const run of data.pluginRuns ?? []) {
    const group = groups.get(run.comparisonId) ?? [];
    group.push(run);
    groups.set(run.comparisonId, group);
  }

  return [...groups.entries()]
    .filter(([, runs]) => (
      runs.some((run) => run.plugin.id === 'vanilla')
      && runs.some((run) => run.plugin.id !== 'vanilla')
    ))
    .map(([comparisonId, runs]) => ({
      comparisonId,
      runs: [...runs].sort((left, right) => (
        Number(right.plugin.id === 'vanilla') - Number(left.plugin.id === 'vanilla')
        || left.plugin.name.localeCompare(right.plugin.name)
      )),
      referenceRun: runs.find((run) => run.plugin.id === 'vanilla') ?? runs[0],
      targets: runs[0].targets,
    }))
    .sort((left, right) => compareRunsByCommit(left.referenceRun, right.referenceRun));
}

export function selectPluginComparison(group, target, suites, baselinePluginId = 'vanilla') {
  const baselineRun = group?.runs.find((run) => run.plugin.id === baselinePluginId) ?? group?.runs[0] ?? null;
  const pluginRuns = group?.runs ?? [];
  const baselineTests = selectedTests(baselineRun, target, suites);
  const baselineById = new Map(baselineTests.map((test) => [test.logicalTestId, test]));

  const rows = baselineTests.map((test) => ({
    test,
    values: pluginRuns.map((run) => {
      const result = selectedTests(run, target, suites)
        .find((candidate) => candidate.logicalTestId === test.logicalTestId) ?? null;
      const baseline = baselineById.get(test.logicalTestId);
      const comparable = completed(result) && completed(baseline) && baseline.durationSeconds !== 0;
      return {
        run,
        result,
        comparable,
        delta: comparable ? ((result.durationSeconds - baseline.durationSeconds) / baseline.durationSeconds) * 100 : null,
        speedup: comparable ? baseline.durationSeconds / result.durationSeconds : null,
      };
    }),
  }));

  const summaries = pluginRuns.map((run) => {
    const tests = selectedTests(run, target, suites);
    const completedTests = tests.filter(completed);
    const comparisons = rows
      .map((row) => row.values.find((value) => value.run.runId === run.runId))
      .filter(Boolean);
    const comparable = comparisons.filter((comparison) => comparison.comparable);
    const fullyComparable = rows.length > 0
      && comparable.length === rows.length
      && completedTests.length === tests.length;
    const geometricMeanRatio = comparable.length > 0
      ? Math.exp(comparable.reduce((sum, comparison) => sum + Math.log(comparison.result.durationSeconds
        / baselineById.get(comparison.result.logicalTestId).durationSeconds), 0) / comparable.length)
      : null;
    const counts = comparable.reduce((result, comparison) => {
      const state = comparison.delta > PLUGIN_NOISE_TOLERANCE
        ? 'slower'
        : comparison.delta < -PLUGIN_NOISE_TOLERANCE ? 'faster' : 'neutral';
      result[state] += 1;
      return result;
    }, { faster: 0, neutral: 0, slower: 0 });

    return {
      run,
      total: tests.length,
      completed: completedTests.length,
      failed: tests.filter((test) => test.status === 'failed').length,
      timeout: tests.filter((test) => test.status === 'timeout').length,
      duration: completedTests.length === tests.length && tests.length > 0 ? durationTotal(completedTests) : null,
      comparable: comparable.length,
      overhead: run.runId === baselineRun?.runId ? 0 : geometricMeanRatio == null ? null : (geometricMeanRatio - 1) * 100,
      estimated: run.runId !== baselineRun?.runId && geometricMeanRatio != null && !fullyComparable,
      counts,
    };
  });

  const findings = pluginRuns.flatMap((run) => selectedTests(run, target, suites).flatMap((test) => (
    (test.findings?.length ? test.findings : test.status !== 'completed' && test.error
      ? [{ type: test.status, summary: test.error }]
      : []).map((finding) => ({ run, test, finding }))
  )));

  return { baselineRun, pluginRuns, rows, summaries, findings };
}
