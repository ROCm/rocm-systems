import { compareRunExecution } from '../../data/runOrdering';

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
    target: current.target ?? 'gfx1250', mode: current.mode ?? 'ST', detail: true };
}

export function selectionForCandidate(current, candidateId, automaticReferenceId) {
  return { ...current, candidateId, referenceId: current.manual ? current.referenceId : automaticReferenceId ?? null };
}

export function resolveBranchSelection(snapshot = {}, { branches = [], runs = [], automaticReferenceId = null }) {
  const branch = snapshot.branch ?? runs.find((run) => run.runId === snapshot.candidateId)?.branch ?? branches[0]?.branch ?? null;
  const branchRuns = runs.filter((run) => run.branch === branch).sort(compareRunExecution).reverse();
  return {
    branch,
    candidateId: snapshot.candidateId ?? branchRuns[0]?.runId ?? null,
    referenceId: snapshot.referenceId ?? (snapshot.manual ? null : automaticReferenceId),
    manual: snapshot.manual ?? false,
    target: snapshot.target ?? 'gfx1250', mode: snapshot.mode ?? 'ST', detail: snapshot.detail ?? false,
  };
}
