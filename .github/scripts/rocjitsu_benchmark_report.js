// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

'use strict';

async function reportBenchmark({ github, context, benchmarkResult, publishResult }) {
  const { owner, repo } = context.repo;
  const number = context.payload.pull_request.number;
  const marker = `<!-- rocjitsu-benchmark-report:${number} -->`;
  const runUrl = `${context.serverUrl}/${owner}/${repo}/actions/runs/${context.runId}`;
  const body = [
    marker,
    '## Rocjitsu benchmark dashboard',
    '',
    publishResult === 'success'
      ? 'Results published to the shared dashboard dataset.'
      : 'Results were not confirmed published for this attempt. Check the workflow run for details; the dashboard may only contain earlier attempts.',
    '',
    `- Benchmark job: **${benchmarkResult}**`,
    `- Publication job: **${publishResult}**`,
    '',
    '[Open the dashboard](https://rocm.github.io/rocm-systems/rocjitsu-dashboard/)',
    '',
    'Left sidebar: **Branch Runs** → **Search branch, PR or SHA**.',
    `Search for **#${number}** to find this PR's published runs.`,
    '',
    `[View the workflow run](${runUrl}).`,
  ].join('\n');
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner, repo, issue_number: number, per_page: 100,
  });
  const existing = comments.find((comment) =>
    comment.user?.login === 'github-actions[bot]' &&
    comment.performed_via_github_app?.slug === 'github-actions' &&
    comment.body?.includes(marker),
  );
  // The reporting job serializes this read/write per PR to avoid duplicate creates.
  if (existing) {
    await github.rest.issues.updateComment({ owner, repo, comment_id: existing.id, body });
  } else {
    await github.rest.issues.createComment({ owner, repo, issue_number: number, body });
  }
}

module.exports = { reportBenchmark };
