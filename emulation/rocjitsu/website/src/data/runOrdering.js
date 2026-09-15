const COMPARISON_SCOPE_FIELDS = ['branch', 'environmentId'];

function parsedTime(value) {
  const time = Date.parse(value);
  return Number.isFinite(time) ? time : Number.NEGATIVE_INFINITY;
}

export function isRunCompleted(run) {
  return Array.isArray(run?.tests)
    && run.tests.length > 0
    && run.tests.every((test) => test.status === 'completed');
}

export function commitShaFor(run) {
  return run?.provenance?.rocjitsuCommitSha ?? '';
}

export function commitTimestampFor(run) {
  return run?.commitTimestamp ?? run?.timestamp;
}

export function sameComparisonScope(left, right) {
  return COMPARISON_SCOPE_FIELDS.every((field) => left?.[field] === right?.[field]);
}

export function compareRunExecution(left, right) {
  return parsedTime(left?.timestamp) - parsedTime(right?.timestamp)
    || String(left?.runId ?? '').localeCompare(String(right?.runId ?? ''));
}

export function compareCommitPosition(left, right) {
  const leftSha = commitShaFor(left);
  const rightSha = commitShaFor(right);
  if (leftSha && leftSha === rightSha) return 0;

  const timeDifference = parsedTime(commitTimestampFor(left)) - parsedTime(commitTimestampFor(right));
  return timeDifference || leftSha.localeCompare(rightSha);
}

export function compareRunsByCommit(left, right) {
  return compareCommitPosition(left, right)
    || compareRunExecution(left, right);
}

export function sortRunsByCommit(runs) {
  return [...runs].sort(compareRunsByCommit);
}

// A backfill is any run executed after a run that tested a newer commit. Walking execution order
// once and tracking the newest commit reached so far answers that for every run, instead of
// comparing each run against the whole history.
export function backfillRunIds(runs) {
  const ids = new Set();
  let newestCommitRun = null;
  [...runs].sort(compareRunExecution).forEach((run) => {
    if (newestCommitRun && compareCommitPosition(run, newestCommitRun) < 0) {
      ids.add(run.runId);
      return;
    }
    newestCommitRun = run;
  });
  return ids;
}
