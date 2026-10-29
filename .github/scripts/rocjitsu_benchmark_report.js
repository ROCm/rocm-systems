// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

'use strict';

async function previousOrder(github, context, body) {
  const identity = body.match(/<!-- rocjitsu-benchmark-report-run:\d+:(\d+):(\d+) -->/);
  if (identity) {
    return { runNumber: BigInt(identity[1]), runAttempt: BigInt(identity[2]) };
  }
  // Pre-ordering comments only stored a run link. Resolve it before replacing
  // them, rather than letting an older completion claim the first upgrade.
  const link = body.match(/\[View the workflow run\]\(([^)]+)\)/);
  if (!link) throw new Error('Existing benchmark report has no workflow run identity.');
  const url = new URL(link[1]);
  const { owner, repo } = context.repo;
  const prefix = `/${owner}/${repo}/actions/runs/`;
  if (url.origin !== new URL(context.serverUrl).origin || !url.pathname.startsWith(prefix)) {
    throw new Error('Existing benchmark report has an unexpected workflow run URL.');
  }
  const run = url.pathname.slice(prefix.length).match(/^(\d+)(?:\/attempts\/(\d+))?$/);
  if (!run) throw new Error('Existing benchmark report has an invalid workflow run URL.');
  const { data } = await github.rest.actions.getWorkflowRun({ owner, repo, run_id: run[1] });
  return { runNumber: BigInt(data.run_number), runAttempt: BigInt(run[2] || data.run_attempt) };
}

async function reportBenchmark({ github, context, benchmarkResult, publishResult, runNumber, runAttempt, headSha }) {
  const { owner, repo } = context.repo;
  const number = context.payload.pull_request.number;
  const marker = `<!-- rocjitsu-benchmark-report:${number} -->`;
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
    const previous = await previousOrder(github, context, existing.body);
    if (BigInt(runNumber) < previous.runNumber ||
        (BigInt(runNumber) === previous.runNumber && BigInt(runAttempt) < previous.runAttempt)) {
      return;
    }
  }
  const dashboardUrl = new URL('https://rocm.github.io/rocm-systems/rocjitsu-dashboard/');
  dashboardUrl.searchParams.set('view', 'branch');
  dashboardUrl.searchParams.set('branch', context.payload.pull_request.head.ref);
  const body = [
    marker,
    `<!-- rocjitsu-benchmark-report-run:${context.runId}:${runNumber}:${runAttempt} -->`,
    '## Rocjitsu benchmark dashboard',
    '',
    publishResult === 'success'
      ? 'Results published to the shared dashboard dataset.'
      : 'Results were not confirmed published for this attempt. Check the workflow run for details; the dashboard may only contain earlier attempts.',
    '',
    `- Measured head: [\`${headSha}\`](${context.serverUrl}/${owner}/${repo}/commit/${headSha})`,
    `- Run: **${runNumber}**, attempt: **${runAttempt}**`,
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
