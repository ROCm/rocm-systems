import { formatRunTime, shortSha } from '../../utils/formatters';
import { commitTimestampFor } from '../../data/runOrdering';

export function runSelectionLabel(run) {
  return `${shortSha(run)} · Run ${formatRunTime(run.timestamp)}`;
}

export function runSearchFields(run) {
  return [
    runSelectionLabel(run),
    `Test run ${formatRunTime(run.timestamp)}`,
    `Commit ${formatRunTime(commitTimestampFor(run))}`,
    run.runId, run.branch, ...(run.modes ?? []), ...(run.targets ?? []),
    run.timestamp, commitTimestampFor(run),
    run.provenance?.rocjitsuCommitSha, run.provenance?.commitMessage,
  ];
}
