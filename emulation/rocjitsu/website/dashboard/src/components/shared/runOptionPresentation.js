import { formatRunTime, shortSha } from '../../utils/formatters';
import { commitTimestampFor } from '../../data/runOrdering';

export function runSelectionLabel(run) {
  return `${shortSha(run)} · Run ${formatRunTime(run.timestamp)}`;
}

// Run Comparison's closed pickers also name the run's branch: its default pair can come from any branch.
export function runSelectionLabelWithBranch(run) {
  return `${runSelectionLabel(run)} · ${run.branch}`;
}

export function runSearchFields(run) {
  return [
    runSelectionLabelWithBranch(run),
    `Test run ${formatRunTime(run.timestamp)}`,
    `Commit ${formatRunTime(commitTimestampFor(run))}`,
    run.runId, run.branch, ...(run.targets ?? []),
    run.timestamp, commitTimestampFor(run),
    run.provenance?.rocjitsuCommitSha, run.provenance?.commitMessage,
  ];
}
