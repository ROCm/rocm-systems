import { commitShaFor, compareRunExecution } from '../../../../scripts/benchmark-data/runOrdering.mjs';
export { commitShaFor, compareRunExecution };
import { isValidTimestamp } from '../utils/timestamp.js';

function parsedTime(value) {
  const time = Date.parse(value);
  return Number.isFinite(time) ? time : Number.NEGATIVE_INFINITY;
}

export function commitTimestampFor(run) {
  return run?.commitTimestamp;
}

export function hasCommitTimestamp(run) {
  return isValidTimestamp(run?.commitTimestamp);
}

export function compareCommitPosition(left, right) {
  const leftSha = commitShaFor(left);
  const rightSha = commitShaFor(right);
  if (leftSha && leftSha === rightSha) return 0;

  const timeDifference = parsedTime(commitTimestampFor(left)) - parsedTime(commitTimestampFor(right));
  return timeDifference || leftSha.localeCompare(rightSha);
}

function compareRunsByCommit(left, right) {
  return compareCommitPosition(left, right)
    || compareRunExecution(left, right);
}

export function sortRunsByCommit(runs) {
  return runs.filter(hasCommitTimestamp).sort(compareRunsByCommit);
}

// A backfill is any run executed after a run that tested a newer commit. Walking execution order
// once and tracking the newest commit reached so far answers that for every run, instead of
// comparing each run against the whole history.
export function backfillRunIds(runs) {
  const ids = new Set();
  let newestCommitRun = null;
  [...runs].sort(compareRunExecution).forEach((run) => {
    if (!hasCommitTimestamp(run)) return;
    if (newestCommitRun && compareCommitPosition(run, newestCommitRun) < 0) {
      ids.add(run.runId);
      return;
    }
    newestCommitRun = run;
  });
  return ids;
}
