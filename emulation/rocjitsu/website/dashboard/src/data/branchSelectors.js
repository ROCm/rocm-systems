import { compareRunExecution, commitShaFor } from './runOrdering.js';
import { selectRunComparison } from './selectors.js';

const finishTime = (run) => Date.parse(run.finishedAt);
const compareFinish = (left, right) => finishTime(left) - finishTime(right) || String(left.runId).localeCompare(String(right.runId));

export function selectAutomaticReference(data, candidate) {
  const develop = (data.runs ?? []).filter((run) => run.branch === (data.canonicalBranch ?? 'develop'));
  if (candidate?.sourceBase?.branch === (data.canonicalBranch ?? 'develop')) {
    const matches = develop.filter((run) => commitShaFor(run) === candidate.sourceBase.commit);
    // Several attempts can share the recorded base commit: use the one that finished last, and a checkpoint (latest start) only when it is the only kind.
    const exact = matches.filter((run) => run.finishedAt != null).sort(compareFinish).at(-1) ?? matches.sort(compareRunExecution).at(-1);
    if (exact) return { run: exact, reason: 'exact-base', description: `Exact develop base ${commitShaFor(exact)} · attempt ${exact.runId}` };
  }
  // The baseline is the latest develop result that existed when the candidate started: finished at or before the candidate's start.
  // A run with no finish record (killed harness, checkpoint) is a partial result and never qualifies.
  const earlier = candidate ? develop.filter((run) => run.finishedAt != null && finishTime(run) <= Date.parse(candidate.timestamp)).sort(compareFinish).at(-1) : null;
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
