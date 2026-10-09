import { compareRunExecution } from '../../data/runOrdering';

export function configurationTargets(runs, selectedTarget = null) {
  return [...new Set(runs.flatMap((run) => (run?.configurations ?? []).map(({ target }) => target))
    .concat(selectedTarget == null ? [] : [selectedTarget]))].sort();
}

function configurationDefaults(snapshot, runs) {
  const preferred = runs.find((run) => run?.configurations?.length);
  const target = snapshot.target ?? configurationTargets(preferred ? [preferred] : runs)[0] ?? null;
  const configurations = runs.find((run) => run?.configurations?.some((c) => c.target === target))?.configurations ?? [];
  const mode = snapshot.mode ?? ['ST', 'MT'].find((value) => configurations.some((c) => c.target === target && c.threadingMode === value)) ?? 'ST';
  return { target, mode };
}

// Atomic snapshots keep route identities and displayed scope together.
export function selectionForAction(current, action) {
  switch (action.type) {
    case 'reference': return { ...current, referenceId: action.runId, manual: true };
    case 'automatic': return { ...current, referenceId: action.runId ?? null, manual: false };
    case 'configuration': return { ...current, target: action.target, mode: action.mode };
    case 'back': return { ...current, detail: false };
    default: return current;
  }
}

export function selectionForBranch(current, entry, referenceId) {
  return { branch: entry.branch, candidateId: entry.latestRun.runId, referenceId: referenceId ?? null, manual: false,
    ...configurationDefaults(current, [entry.latestRun]), detail: true };
}

export function selectionForCandidate(current, candidateId, automaticReferenceId) {
  return { ...current, candidateId, referenceId: current.manual ? current.referenceId : automaticReferenceId ?? null };
}

export function resolveBranchSelection(snapshot = {}, { branches = [], runs = [], automaticReferenceId = null }) {
  const branch = snapshot.branch ?? runs.find((run) => run.runId === snapshot.candidateId)?.branch ?? branches[0]?.branch ?? null;
  const branchRuns = runs.filter((run) => run.branch === branch).sort(compareRunExecution).reverse();
  const candidateId = snapshot.candidateId ?? branchRuns[0]?.runId ?? null;
  const referenceId = snapshot.referenceId ?? (snapshot.manual ? null : automaticReferenceId);
  return {
    branch, candidateId, referenceId,
    manual: snapshot.manual ?? false,
    ...configurationDefaults(snapshot, [runs.find((run) => run.runId === candidateId), runs.find((run) => run.runId === referenceId), ...runs]), detail: snapshot.detail ?? false,
  };
}
