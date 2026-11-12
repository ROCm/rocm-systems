// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

function parsedTime(value) {
  const time = Date.parse(value);
  return Number.isFinite(time) ? time : Number.NEGATIVE_INFINITY;
}

export function commitShaFor(run) {
  const sha = run?.provenance?.rocjitsuCommitSha;
  return typeof sha === 'string' ? sha : '';
}

export function compareRunExecution(left, right) {
  return parsedTime(left?.timestamp) - parsedTime(right?.timestamp)
    || String(left?.runId ?? '').localeCompare(String(right?.runId ?? ''));
}

export function findRecordedBase(runs, candidate, canonicalBranch = 'develop') {
  if (candidate?.sourceBase?.branch !== canonicalBranch) return null;
  return runs.filter((run) => run.branch === canonicalBranch
    && commitShaFor(run) === candidate.sourceBase.commit
    && commitShaFor(run) !== commitShaFor(candidate))
    .sort(compareRunExecution).at(-1) ?? null;
}
