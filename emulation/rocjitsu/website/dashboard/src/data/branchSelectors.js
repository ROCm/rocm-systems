import { compareRunExecution, commitShaFor } from './runOrdering.js';
import { selectRunComparison } from './selectors.js';

export function selectAutomaticReference(data, candidate) {
  const develop = (data.runs ?? []).filter((run) => run.branch === (data.canonicalBranch ?? 'develop'));
  if (candidate?.sourceBase?.branch === (data.canonicalBranch ?? 'develop')) {
    const exact = develop.filter((run) => commitShaFor(run) === candidate.sourceBase.commit).sort(compareRunExecution).at(-1);
    if (exact) return { run: exact, reason: 'exact-base', description: `Exact develop base ${commitShaFor(exact)} · attempt ${exact.runId}` };
  }
  const earlier = candidate ? develop.filter((run) => Date.parse(run.timestamp) < Date.parse(candidate.timestamp)).sort(compareRunExecution).at(-1) : null;
  return earlier ? { run: earlier, reason: 'earlier-develop', description: `Latest develop execution completed before this candidate · attempt ${earlier.runId}` }
    : { run: null, reason: 'unavailable', description: 'No published develop reference is available for this candidate.' };
}

export function selectConfigurationComparison(candidate, baseline, { target, mode, suites, query = '' }) {
  const search = query.trim().toLowerCase();
  const filterRun = (run) => run ? { ...run, tests: run.tests.filter((test) => !search || [test.name, test.logicalTestId, test.suite].some((value) => value.toLowerCase().includes(search))) } : null;
  const result = selectRunComparison(filterRun(candidate), filterRun(baseline), { targets: [target], modes: [mode], suites });
  return { ...result, comparable: [...result.comparable].sort((a, b) => Math.abs(b.deltaSeconds) - Math.abs(a.deltaSeconds)
    || a.candidateTest.testId.localeCompare(b.candidateTest.testId)) };
}

export function selectPublishedBranches(data, { query = '', pr = 'all' } = {}) {
  const end = Date.parse(data.generatedAt); const start = end - 30 * 86400_000;
  const groups = new Map(); const search = query.trim().toLowerCase();
  for (const run of data.allRuns ?? []) {
    const time = Date.parse(run.timestamp);
    if (run.branch === data.canonicalBranch || !Number.isFinite(time) || time > end) continue;
    const group = groups.get(run.branch) ?? [];
    group.push(run); groups.set(run.branch, group);
  }
  return [...groups].map(([branch, group]) => {
    const runs = [...group].sort(compareRunExecution).reverse();
    return { branch, runs, latestRun: runs[0], pullRequest: runs[0].pullRequest ?? null };
  }).filter(({ branch, runs, pullRequest }) => {
    if (Date.parse(runs[0].timestamp) < start) return false;
    if (['with-pr', 'with'].includes(pr) && !pullRequest) return false;
    if (['no-pr', 'without-pr', 'without'].includes(pr) && pullRequest) return false;
    return !search || branch.toLowerCase().includes(search) || runs.some((run) => commitShaFor(run).toLowerCase().includes(search)
      || String(run.pullRequest?.number ?? '').includes(search.replace(/^#/, '')));
  }).sort((a, b) => compareRunExecution(b.latestRun, a.latestRun));
}
