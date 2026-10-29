// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { reportBenchmark } = require('../rocjitsu_benchmark_report.js');

function fixture() {
  const comments = [];
  const writes = [];
  const github = {
    rest: { issues: {
      listComments: Symbol('listComments'),
      async createComment(args) {
        writes.push(['create', args]);
        comments.push({ id: 7, body: args.body, user: { login: 'github-actions[bot]' },
          performed_via_github_app: { slug: 'github-actions' } });
      },
      async updateComment(args) {
        writes.push(['update', args]);
        comments.find((comment) => comment.id === args.comment_id).body = args.body;
      },
    } },
    async paginate(method, args) {
      assert.equal(method, github.rest.issues.listComments);
      assert.deepEqual(args, { owner: 'ROCm', repo: 'rocm-systems', issue_number: 42, per_page: 100 });
      return comments;
    },
  };
  const args = {
    github,
    context: { repo: { owner: 'ROCm', repo: 'rocm-systems' },
      payload: { pull_request: { number: 42, head: { ref: 'feature/source' } } },
      serverUrl: 'https://github.com', runId: 123 },
    runNumber: '10', runAttempt: '1', headSha: 'a'.repeat(40),
    benchmarkResult: 'success', publishResult: 'success',
  };
  return { args, comments, writes };
}

test('creates one PR report with direct dashboard and workflow links', async () => {
  const { args, comments, writes } = fixture();
  await reportBenchmark(args);
  assert.equal(writes.length, 1);
  assert.equal(writes[0][0], 'create');
  assert.equal(writes[0][1].issue_number, 42);
  const body = comments[0].body;
  assert.match(body, /<!-- rocjitsu-benchmark-report:42 -->/);
  assert.match(body, /Results published/);
  assert.match(body, /https:\/\/rocm.github.io\/rocm-systems\/rocjitsu-dashboard\//);
  const dashboardLink = body.match(/\[Open the dashboard\]\(([^)]+)\)/)[1];
  const dashboardUrl = new URL(dashboardLink);
  assert.equal(dashboardUrl.searchParams.get('view'), 'branch');
  assert.equal(dashboardUrl.searchParams.get('branch'), 'feature/source');
  assert.match(body, /\[Open the dashboard\]\([^\n]+\)\n\n\[View the workflow run\]/);
  assert.match(body, /https:\/\/github.com\/ROCm\/rocm-systems\/actions\/runs\/123/);
});

test('later runs update the same comment instead of adding another', async () => {
  const { args, comments, writes } = fixture();
  await reportBenchmark(args);
  await reportBenchmark({ ...args, publishResult: 'failure' });
  assert.equal(comments.length, 1);
  assert.deepEqual(writes.map(([kind]) => kind), ['create', 'update']);
  assert.equal(writes[1][1].comment_id, 7);
  assert.match(comments[0].body, /Results were not confirmed published/);
});

for (const publishResult of ['failure', 'skipped', 'cancelled']) {
  test(`does not claim publication when publisher is ${publishResult}`, async () => {
    const { args, comments } = fixture();
    await reportBenchmark({ ...args, benchmarkResult: 'failure', publishResult });
    assert.match(comments[0].body, /Results were not confirmed published/);
    assert.match(comments[0].body, /Benchmark job: \*\*failure\*\*/);
    assert.match(comments[0].body, new RegExp(`Publication job: \\*\\*${publishResult}\\*\\*`));
    assert.doesNotMatch(comments[0].body, /Results published/);
  });
}

test('published failure results do not claim benchmark success', async () => {
  const { args, comments } = fixture();
  await reportBenchmark({ ...args, benchmarkResult: 'failure' });
  assert.match(comments[0].body, /Results published/);
  assert.match(comments[0].body, /Benchmark job: \*\*failure\*\*/);
});

test('does not overwrite a user comment or a different report', async () => {
  const { args, comments, writes } = fixture();
  comments.push({ id: 1, user: { login: 'someone' }, body: '<!-- rocjitsu-benchmark-report:42 -->' });
  comments.push({ id: 2, user: { login: 'github-actions[bot]' },
    performed_via_github_app: { slug: 'github-actions' }, body: '<!-- rocjitsu-presubmit-fuzz:42 -->' });
  await reportBenchmark(args);
  assert.equal(writes[0][0], 'create');
  assert.equal(comments.length, 3);
});

test('newer failure report cannot be replaced by an older successful run', async () => {
  const { args, comments, writes } = fixture();
  await reportBenchmark({ ...args, runNumber: '20', context: { ...args.context, runId: 200 },
    headSha: 'b'.repeat(40), benchmarkResult: 'failure', publishResult: 'skipped' });
  const newestBody = comments[0].body;
  await reportBenchmark({ ...args, runNumber: '10', context: { ...args.context, runId: 100 } });
  assert.equal(writes.length, 1);
  assert.equal(comments[0].body, newestBody);
  assert.match(newestBody, /b{40}/);
  assert.match(newestBody, /Run: \*\*20\*\*, attempt: \*\*1\*\*/);
});

test('reruns order attempts within a run without superseding a newer run', async () => {
  const { args, comments, writes } = fixture();
  await reportBenchmark(args);
  await reportBenchmark({ ...args, runAttempt: '2', publishResult: 'failure' });
  assert.equal(writes.length, 2);
  assert.match(comments[0].body, /attempt: \*\*2\*\*/);
  const latest = comments[0].body;
  await reportBenchmark(args);
  await reportBenchmark({ ...args, runNumber: '9', runAttempt: '99' });
  assert.equal(comments[0].body, latest);
  assert.equal(writes.length, 2);
});

test('upgrades a legacy comment only after checking its workflow run identity', async () => {
  const { args, comments, writes } = fixture();
  comments.push({ id: 9, user: { login: 'github-actions[bot]' },
    performed_via_github_app: { slug: 'github-actions' },
    body: '<!-- rocjitsu-benchmark-report:42 -->\n[View the workflow run](https://github.com/ROCm/rocm-systems/actions/runs/200).' });
  args.github.rest.actions = { async getWorkflowRun(request) {
    assert.deepEqual(request, { owner: 'ROCm', repo: 'rocm-systems', run_id: '200' });
    return { data: { run_number: 20, run_attempt: 2 } };
  } };
  const legacy = comments[0].body;
  await reportBenchmark(args);
  assert.equal(comments[0].body, legacy);
  assert.equal(writes.length, 0);
  await reportBenchmark({ ...args, runNumber: '21' });
  assert.equal(writes.length, 1);
  assert.equal(writes[0][0], 'update');
  assert.equal(writes[0][1].comment_id, 9);
  assert.match(comments[0].body, /<!-- rocjitsu-benchmark-report-run:123:21:1 -->/);
});

test('legacy lookup failure leaves the existing report untouched', async () => {
  const { args, comments, writes } = fixture();
  await reportBenchmark(args);
  comments[0].body = comments[0].body.replace(/<!-- rocjitsu-benchmark-report-run:.*? -->\n/, '');
  args.github.rest.actions = { async getWorkflowRun() { throw new Error('lookup failed'); } };
  await assert.rejects(reportBenchmark(args), /lookup failed/);
  assert.equal(writes.length, 1);
});

test('report deep link preserves branch names as URL data', async () => {
  const { args, comments } = fixture();
  args.context.payload.pull_request.head.ref = 'users/name/topic&view=compare#fragment';
  await reportBenchmark(args);
  const link = comments[0].body.match(/\[Open the dashboard\]\(([^)]+)\)/)[1];
  const url = new URL(link);
  assert.equal(url.searchParams.get('view'), 'branch');
  assert.equal(url.searchParams.get('branch'), 'users/name/topic&view=compare#fragment');
  assert.equal(url.hash, '');
});

test('API errors fail reporting rather than claiming delivery', async () => {
  const { args } = fixture();
  args.github.rest.issues.createComment = async () => { throw new Error('API unavailable'); };
  await assert.rejects(reportBenchmark(args), /API unavailable/);
});
