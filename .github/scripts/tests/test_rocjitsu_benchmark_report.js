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
      payload: { pull_request: { number: 42 } },
      serverUrl: 'https://github.com', runId: 123 },
    benchmarkResult: 'success', publishResult: 'success',
  };
  return { args, comments, writes };
}

test('creates one PR report with dashboard link and navigation instructions', async () => {
  const { args, comments, writes } = fixture();
  await reportBenchmark(args);
  assert.equal(writes.length, 1);
  assert.equal(writes[0][0], 'create');
  assert.equal(writes[0][1].issue_number, 42);
  const body = comments[0].body;
  assert.match(body, /<!-- rocjitsu-benchmark-report:42 -->/);
  assert.match(body, /Results published/);
  assert.match(body, /https:\/\/rocm.github.io\/rocm-systems\/rocjitsu-dashboard\//);
  assert.match(body, /Left sidebar: \*\*Branch Runs\*\* → \*\*Search branch, PR or SHA\*\*/);
  assert.match(body, /Search for \*\*#42\*\*/);
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

test('API errors fail reporting rather than claiming delivery', async () => {
  const { args } = fixture();
  args.github.rest.issues.createComment = async () => { throw new Error('API unavailable'); };
  await assert.rejects(reportBenchmark(args), /API unavailable/);
});
