// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

'use strict';

function previousOrder(body) {
  const identity = body.match(/<!-- rocjitsu-benchmark-report-run:\d+:(\d+):(\d+) -->/);
  if (!identity) throw new Error('Existing benchmark report has no workflow run identity.');
  return { runNumber: BigInt(identity[1]), runAttempt: BigInt(identity[2]) };
}

async function reportBenchmark({
  github,
  context,
  core,
  benchmarkResult,
  publishResult,
  runNumber,
  runAttempt,
  benchmarkAttempt,
  publishedRunId,
  headSha,
}) {
  const { owner, repo } = context.repo;
  let pull = context.payload.pull_request;
  const postCommit = context.eventName === 'push';
  if (postCommit) {
    if (context.ref !== 'refs/heads/develop' || headSha !== context.sha) return;
    const associated = await github.paginate(
      github.rest.repos.listPullRequestsAssociatedWithCommit,
      { owner, repo, commit_sha: headSha, per_page: 100 },
    );
    const matches = associated.filter((candidate) =>
      candidate.base?.ref === 'develop' &&
      candidate.base?.repo?.full_name === `${owner}/${repo}` &&
      candidate.merged_at && candidate.merge_commit_sha === headSha,
    );
    if (matches.length !== 1) {
      core?.warning(`Expected one associated develop PR for ${headSha}, found ${matches.length}.`);
      return;
    }
    [pull] = matches;
  }
  if (!pull) return;
  const number = pull.number;
  const marker = postCommit
    ? `<!-- rocjitsu-post-commit-benchmark:${headSha} -->`
    : `<!-- rocjitsu-benchmark-report:${number} -->`;
  const runUrl = `${context.serverUrl}/${owner}/${repo}/actions/runs/${context.runId}/attempts/${runAttempt}`;
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner, repo, issue_number: number, per_page: 100,
  });
  const existing = comments.find((comment) =>
    comment.user?.login === 'github-actions[bot]' &&
    comment.performed_via_github_app?.slug === 'github-actions' &&
    comment.body?.includes(marker),
  );
  if (existing) {
    const previous = previousOrder(existing.body);
    if (BigInt(runNumber) < previous.runNumber ||
        (BigInt(runNumber) === previous.runNumber && BigInt(runAttempt) < previous.runAttempt)) {
      return;
    }
  }
  const dashboardUrl = new URL('https://rocm.github.io/rocm-systems/rocjitsu-dashboard/');
  dashboardUrl.searchParams.set('view', 'branch');
  dashboardUrl.searchParams.set('branch', postCommit ? 'develop' : pull.head.ref);
  dashboardUrl.searchParams.set('detail', '1');
  if (publishResult === 'success') {
    if (!publishedRunId) throw new Error('Successful publication has no published run ID.');
    dashboardUrl.searchParams.set('run', publishedRunId);
  }
  const body = [
    marker,
    `<!-- rocjitsu-benchmark-report-run:${context.runId}:${runNumber}:${runAttempt} -->`,
    postCommit ? '## Rocjitsu post-merge benchmark results' : '## Rocjitsu benchmark dashboard',
    '',
    ...(benchmarkResult === 'cancelled' ? [
      '**Benchmark cancelled. This attempt did not complete.**',
      '',
      postCommit
        ? 'Rerun this workflow to measure the merged commit.'
        : "To measure the current code, rerun the workflow for the current PR head, not an older commit's run.",
      '',
    ] : []),
    publishResult === 'success'
      ? 'Results published to the shared dashboard dataset.'
      : 'Results were not confirmed published for this attempt. Check the workflow run for details; the dashboard may only contain earlier attempts.',
    '',
    `- ${benchmarkResult === 'cancelled' ? 'Selected commit' : postCommit ? 'Measured commit' : 'Measured head'}: [\`${headSha}\`](${context.serverUrl}/${owner}/${repo}/commit/${headSha})`,
    `- Workflow run: [#${runNumber}](${runUrl})`,
    `- Reporting attempt: **${runAttempt}**`,
    `- Benchmark attempt: **${benchmarkAttempt || 'not recorded'}**`,
    `- Benchmark job: **${benchmarkResult}**`,
    `- Publication job: **${publishResult}**`,
    '',
    `[Open the dashboard](${dashboardUrl.href})`,
    '',
    `[View the workflow run](${runUrl}).`,
  ].join('\n');
  // The reporting job serializes the read/compare/write per PR.
  if (existing) {
    await github.rest.issues.updateComment({ owner, repo, comment_id: existing.id, body });
  } else {
    await github.rest.issues.createComment({ owner, repo, issue_number: number, body });
  }
}

module.exports = { reportBenchmark };
