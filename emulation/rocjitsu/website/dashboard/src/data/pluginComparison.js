import { compareRunsByCommit } from './runOrdering';

export const PLUGIN_NOISE_TOLERANCE = 3;

function completed(test) {
  return test?.status === 'completed' && Number.isFinite(test.durationSeconds);
}

function selectedTests(run, target, suites, modes) {
  return (run?.tests ?? []).filter((test) => test.target === target && suites.includes(test.suite) && modes.includes(test.mode));
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

export function selectPluginComparison(group, target, suites, baselinePluginId = 'vanilla', modes = ['ST', 'MT']) {
  const baselineRun = group?.runs.find((run) => run.plugin.id === baselinePluginId) ?? group?.runs[0] ?? null;
  const pluginRuns = group?.runs ?? [];
  const baselineTests = selectedTests(baselineRun, target, suites, modes);
  const baselineById = new Map(baselineTests.map((test) => [test.testId, test]));

  const rows = baselineTests.map((test) => ({
    test,
    values: pluginRuns.map((run) => {
      const result = selectedTests(run, target, suites, modes)
        .find((candidate) => candidate.testId === test.testId) ?? null;
      const baseline = baselineById.get(test.testId);
      const comparable = completed(result) && completed(baseline);
      return {
        run,
        result,
        comparable,
        delta: comparable && baseline.durationSeconds > 0 ? ((result.durationSeconds - baseline.durationSeconds) / baseline.durationSeconds) * 100 : null,
      };
    }),
  }));

  const summaries = pluginRuns.map((run) => {
    const tests = selectedTests(run, target, suites, modes);
    const completedTests = tests.filter(completed);
    const comparisons = rows
      .map((row) => row.values.find((value) => value.run.runId === run.runId))
      .filter(Boolean);
    const comparable = comparisons.filter((comparison) => comparison.comparable);
    const ratios = comparable.filter((comparison) => comparison.delta !== null);
    const fullyComparable = rows.length > 0
      && ratios.length === rows.length
      && completedTests.length === tests.length;
    const geometricMeanRatio = ratios.length > 0
      ? Math.exp(ratios.reduce((sum, comparison) => sum + Math.log(comparison.result.durationSeconds
        / baselineById.get(comparison.result.testId).durationSeconds), 0) / ratios.length)
      : null;
    const counts = comparable.reduce((result, comparison) => {
      const state = comparison.delta === null ? 'unavailable' : comparison.delta > PLUGIN_NOISE_TOLERANCE
        ? 'slower'
        : comparison.delta < -PLUGIN_NOISE_TOLERANCE ? 'faster' : 'neutral';
      result[state] += 1;
      return result;
    }, { faster: 0, neutral: 0, slower: 0, unavailable: 0 });

    return {
      run,
      total: tests.length,
      completed: completedTests.length,
      failed: tests.filter((test) => test.status === 'failed').length,
      timeout: tests.filter((test) => test.status === 'timeout').length,
      duration: completedTests.length === tests.length && tests.length > 0 ? durationTotal(completedTests) : null,
      comparable: comparable.length,
      overhead: geometricMeanRatio == null ? null : run.runId === baselineRun?.runId ? 0 : (geometricMeanRatio - 1) * 100,
      estimated: run.runId !== baselineRun?.runId && geometricMeanRatio != null && !fullyComparable,
      counts,
    };
  });

  const errors = pluginRuns.flatMap((run) => selectedTests(run, target, suites, modes)
    .filter((test) => test.status !== 'completed' && test.error)
    .map((test) => ({ run, test, error: test.error })));

  return { baselineRun, pluginRuns, rows, summaries, errors };
}
