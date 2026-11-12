// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { reportBenchmark, buildBenchmarkSummary } = require('../rocjitsu_benchmark_report.js');

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
    runNumber: '10', runAttempt: '1', publishedRunId: 'github-123-attempt-1', headSha: 'a'.repeat(40),
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
  assert.match(body, /## rocjitsu benchmarks: success/);
  assert.match(body, /https:\/\/rocm.github.io\/rocm-systems\/rocjitsu-dashboard\//);
  const dashboardLink = body.match(/\[Dashboard\]\(([^)]+)\)/)[1];
  const dashboardUrl = new URL(dashboardLink);
  assert.equal(dashboardUrl.searchParams.get('view'), 'branch');
  assert.equal(dashboardUrl.searchParams.get('branch'), 'feature/source');
  assert.equal(dashboardUrl.searchParams.get('run'), 'github-123-attempt-1');
  assert.equal(dashboardUrl.searchParams.get('detail'), '1');
  assert.match(body, /\[Dashboard\]\([^\n]+\) · \[View the workflow run\]/);
  assert.match(body, /https:\/\/github.com\/ROCm\/rocm-systems\/actions\/runs\/123/);
});

test('successful report selects the publisher ID without reconstructing it', async () => {
  const { args, comments } = fixture();
  await reportBenchmark({ ...args, publishedRunId: 'publisher-owned-id' });
  const url = new URL(comments[0].body.match(/\[Dashboard\]\(([^)]+)\)/)[1]);
  assert.equal(url.searchParams.get('run'), 'publisher-owned-id');
  assert.equal(url.searchParams.get('detail'), '1');
});

for (const publishedRunId of [undefined, '']) {
  test(`rejects successful publication without an ID (${publishedRunId})`, async () => {
    const { args, writes } = fixture();
    await assert.rejects(reportBenchmark({ ...args, publishedRunId }), /published run ID/);
    assert.equal(writes.length, 0);
  });
}

test('report-only reruns link the original published benchmark attempt', async () => {
  const { args, comments } = fixture();
  await reportBenchmark({ ...args, runAttempt: '3', publishedRunId: 'github-123-attempt-1' });
  const body = comments[0].body;
  const url = new URL(body.match(/\[Dashboard\]\(([^)]+)\)/)[1]);
  assert.equal(url.searchParams.get('run'), 'github-123-attempt-1');
  assert.equal(url.searchParams.get('detail'), '1');
  assert.match(body, /actions\/runs\/123\/attempts\/3/);
});

test('reports benchmark failure without attempt metadata', async () => {
  const { args, comments } = fixture();
  await reportBenchmark({ ...args, benchmarkResult: 'failure',
    publishResult: 'skipped', publishedRunId: '' });
  assert.match(comments[0].body, /benchmarks: failure/);
  assert.doesNotMatch(comments[0].body, /Reporting attempt|Benchmark attempt|Benchmark job|Publication job/);
});

test('cancelled benchmark reports its selected commit and recovery without outputs', async () => {
  const { args, comments } = fixture();
  await reportBenchmark({ ...args, benchmarkResult: 'cancelled', publishResult: 'skipped',
    publishedRunId: '' });
  const body = comments[0].body;
  assert.match(body, /## rocjitsu benchmarks: cancelled/);
  assert.match(body, /Presubmit for commit:?[\s\S]*a{40}/);
  assert.doesNotMatch(body, /Measured head:/);
  assert.doesNotMatch(body, /Reporting attempt|Benchmark attempt|Publication job|Benchmark job/);
  assert.match(body, /Results were not published/);
  const url = new URL(body.match(/\[Dashboard\]\(([^)]+)\)/)[1]);
  assert.equal(url.searchParams.get('branch'), 'feature/source');
  assert.equal(url.searchParams.has('run'), false);
  assert.equal(url.searchParams.get('detail'), '1');
});

test('later runs update the same comment instead of adding another', async () => {
  const { args, comments, writes } = fixture();
  await reportBenchmark(args);
  await reportBenchmark({ ...args, publishResult: 'failure' });
  assert.equal(comments.length, 1);
  assert.deepEqual(writes.map(([kind]) => kind), ['create', 'update']);
  assert.equal(writes[1][1].comment_id, 7);
  assert.match(comments[0].body, /Results were not published/);
});

for (const publishResult of ['failure', 'skipped', 'cancelled']) {
  test(`does not claim publication when publisher is ${publishResult}`, async () => {
    const { args, comments } = fixture();
    await reportBenchmark({ ...args, benchmarkResult: 'failure', publishResult });
    assert.match(comments[0].body, /Results were not published/);
    assert.match(comments[0].body, /## rocjitsu benchmarks: failure/);
    assert.doesNotMatch(comments[0].body, /benchmarks: success/);
    const url = new URL(comments[0].body.match(/\[Dashboard\]\(([^)]+)\)/)[1]);
    assert.equal(url.searchParams.get('branch'), 'feature/source');
    assert.equal(url.searchParams.has('run'), false);
    assert.equal(url.searchParams.get('detail'), '1');
  });
}

for (const publishResult of ['failure', 'skipped', 'cancelled']) {
  test(`keeps a branch-only link without a publisher ID when ${publishResult}`, async () => {
    const { args, comments } = fixture();
    await reportBenchmark({ ...args, publishResult, publishedRunId: '' });
    const url = new URL(comments[0].body.match(/\[Dashboard\]\(([^)]+)\)/)[1]);
    assert.equal(url.searchParams.get('branch'), 'feature/source');
    assert.equal(url.searchParams.has('run'), false);
    assert.equal(url.searchParams.get('detail'), '1');
    assert.match(comments[0].body, /Results were not published/);
  });
}

test('published failure results do not claim benchmark success', async () => {
  const { args, comments } = fixture();
  await reportBenchmark({ ...args, benchmarkResult: 'failure' });
  assert.match(comments[0].body, /run=github-123-attempt-1/);
  assert.match(comments[0].body, /## rocjitsu benchmarks: failure/);
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
});

for (const newerResult of ['cancelled', 'success']) {
  for (const newestFirst of [false, true]) {
    test(`newer ${newerResult} report wins across cancellation (newest first: ${newestFirst})`, async () => {
      const { args, comments, writes } = fixture();
      const newer = { ...args, runNumber: '20', context: { ...args.context, runId: 200 },
        headSha: 'b'.repeat(40), benchmarkResult: newerResult,
        publishResult: newerResult === 'success' ? 'success' : 'skipped',
        publishedRunId: newerResult === 'success' ? 'published-newer' : '' };
      const older = { ...args, runNumber: '10', runAttempt: '99',
        benchmarkResult: newerResult === 'success' ? 'cancelled' : 'success',
        publishResult: newerResult === 'success' ? 'skipped' : 'success' };
      for (const report of newestFirst ? [newer, older] : [older, newer]) {
        await reportBenchmark(report);
      }
      assert.equal(comments.length, 1);
      assert.equal(writes.length, newestFirst ? 1 : 2);
      assert.match(comments[0].body, /rocjitsu-benchmark-report-run:200:20:1/);
      assert.match(comments[0].body, /b{40}/);
      assert.match(comments[0].body, new RegExp(`benchmarks: ${newerResult}`));
    });
  }
}

test('reruns order attempts within a run without superseding a newer run', async () => {
  const { args, comments, writes } = fixture();
  await reportBenchmark(args);
  await reportBenchmark({ ...args, runAttempt: '2', publishResult: 'failure' });
  assert.equal(writes.length, 2);
  const latest = comments[0].body;
  await reportBenchmark(args);
  await reportBenchmark({ ...args, runNumber: '9', runAttempt: '99' });
  assert.equal(comments[0].body, latest);
  assert.equal(writes.length, 2);
});

for (const link of ['', '\n[View the workflow run](https://github.com/ROCm/rocm-systems/actions/runs/200).']) {
  test(`rejects an existing report without identity (workflow link: ${Boolean(link)})`, async () => {
    const { args, comments, writes } = fixture();
    const body = '<!-- rocjitsu-benchmark-report:42 -->' + link;
    comments.push({ id: 9, user: { login: 'github-actions[bot]' },
      performed_via_github_app: { slug: 'github-actions' }, body });
    await assert.rejects(reportBenchmark(args), /no workflow run identity/);
    assert.equal(comments[0].body, body);
    assert.equal(writes.length, 0);
  });
}

test('report deep link preserves branch names as URL data', async () => {
  const { args, comments } = fixture();
  args.context.payload.pull_request.head.ref = 'users/name/topic&view=compare#fragment';
  await reportBenchmark(args);
  const link = comments[0].body.match(/\[Dashboard\]\(([^)]+)\)/)[1];
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

function postCommitFixture() {
  const result = fixture();
  const { args } = result;
  const associated = [{ number: 42, merged_at: '2026-11-11T00:00:00Z',
    merge_commit_sha: args.headSha,
    base: { ref: 'develop', repo: { full_name: 'ROCm/rocm-systems' } } }];
  const warnings = [];
  args.core = { warning(message) { warnings.push(message); } };
  args.context = { ...args.context, payload: {}, eventName: 'push',
    sha: args.headSha, ref: 'refs/heads/develop' };
  args.github.rest.repos = { listPullRequestsAssociatedWithCommit: Symbol('associated') };
  const paginateComments = args.github.paginate;
  args.github.paginate = async (method, parameters) => {
    if (method === args.github.rest.repos.listPullRequestsAssociatedWithCommit) {
      assert.deepEqual(parameters, { owner: 'ROCm', repo: 'rocm-systems',
        commit_sha: args.headSha, per_page: 100 });
      return associated;
    }
    return paginateComments(method, parameters);
  };
  return { ...result, associated, warnings };
}

test('post-merge report finds the exact merged PR and links its develop result', async () => {
  const { args, comments, writes } = postCommitFixture();
  await reportBenchmark(args);
  assert.equal(writes[0][1].issue_number, 42);
  const body = comments[0].body;
  assert.match(body, /<!-- rocjitsu-post-commit-benchmark:a{40} -->/);
  assert.match(body, /rocjitsu benchmarks: success/);
  assert.match(body, /Post-merge for commit[\s\S]*a{40}/);
  const url = new URL(body.match(/\[Dashboard\]\(([^)]+)\)/)[1]);
  assert.equal(url.searchParams.get('branch'), 'develop');
  assert.equal(url.searchParams.get('run'), args.publishedRunId);
  assert.equal(url.searchParams.get('detail'), '1');
});

test('post-merge reruns update their comment without touching the presubmit report', async () => {
  const { args, comments, writes } = postCommitFixture();
  comments.push({ id: 8, user: { login: 'github-actions[bot]' },
    performed_via_github_app: { slug: 'github-actions' },
    body: '<!-- rocjitsu-benchmark-report:42 -->' });
  await reportBenchmark(args);
  await reportBenchmark({ ...args, runAttempt: '2' });
  assert.deepEqual(writes.map(([kind]) => kind), ['create', 'update']);
  assert.equal(comments.length, 2);
  assert.equal(comments[0].body, '<!-- rocjitsu-benchmark-report:42 -->');
});

test('an older post-merge run cannot replace a newer report', async () => {
  const { args, writes } = postCommitFixture();
  await reportBenchmark({ ...args, runAttempt: '2' });
  await reportBenchmark(args);
  assert.equal(writes.length, 1);
});

for (const mismatch of [
  { merged_at: null },
  { merge_commit_sha: 'b'.repeat(40) },
  { base: { ref: 'release', repo: { full_name: 'ROCm/rocm-systems' } } },
  { base: { ref: 'develop', repo: { full_name: 'other/rocm-systems' } } },
]) {
  test(`does not comment on a loosely associated PR: ${JSON.stringify(mismatch)}`, async () => {
    const { args, associated, writes, warnings } = postCommitFixture();
    Object.assign(associated[0], mismatch);
    await reportBenchmark(args);
    assert.equal(writes.length, 0);
    assert.equal(warnings.length, 1);
  });
}

for (const count of [0, 2]) {
  test(`skips ambiguous or missing merged PRs (${count} matches)`, async () => {
    const { args, associated, writes, warnings } = postCommitFixture();
    if (count === 0) associated.length = 0;
    else associated.push({ ...associated[0], number: 43 });
    await reportBenchmark(args);
    assert.equal(writes.length, 0);
    assert.equal(warnings.length, 1);
  });
}

for (const changes of [{ ref: 'refs/heads/feature' }, { sha: 'b'.repeat(40) }]) {
  test(`rejects a push outside the measured develop commit: ${JSON.stringify(changes)}`, async () => {
    const { args, writes } = postCommitFixture();
    args.context = { ...args.context, ...changes };
    args.github.paginate = async () => { assert.fail('Should not query unrelated commits'); };
    await reportBenchmark(args);
    assert.equal(writes.length, 0);
  });
}

test('failed post-merge publication does not link an earlier run as the result', async () => {
  const { args, comments } = postCommitFixture();
  await reportBenchmark({ ...args, benchmarkResult: 'failure', publishResult: 'failure', publishedRunId: '' });
  const body = comments[0].body;
  assert.match(body, /Results were not published/);
  assert.match(body, /## rocjitsu benchmarks: failure/);
  const url = new URL(body.match(/\[Dashboard\]\(([^)]+)\)/)[1]);
  assert.equal(url.searchParams.get('branch'), 'develop');
  assert.equal(url.searchParams.has('run'), false);
});

test('cancelled post-merge report identifies its commit', async () => {
  const { args, comments } = postCommitFixture();
  await reportBenchmark({ ...args, benchmarkResult: 'cancelled', publishResult: 'skipped', publishedRunId: '' });
  assert.match(comments[0].body, /benchmarks: cancelled/);
  assert.match(comments[0].body, /Post-merge for commit/);
  assert.doesNotMatch(comments[0].body, /current PR head/);
});

async function summaryArtifact(t, content) {
  const { mkdtemp, writeFile, rm } = require('node:fs/promises');
  const { tmpdir } = require('node:os');
  const { join } = require('node:path');
  const directory = await mkdtemp(join(tmpdir(), 'rocjitsu-report-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const path = join(directory, 'summary.json');
  await writeFile(path, content);
  return path;
}

test('includes an identified summary for the exact published run', async (t) => {
  const { args, comments } = postCommitFixture();
  const summaryPath = await summaryArtifact(t, JSON.stringify({ runId: args.publishedRunId,
    headSha: args.headSha, markdown: '### Benchmark summary\n\nExample: 4 faster, 2 slower.' }));
  await reportBenchmark({ ...args, summaryPath });
  assert.match(comments[0].body, /Example: 4 faster, 2 slower/);
});

for (const changes of [ { runId: 'old-run' }, { headSha: 'b'.repeat(40) },
  { markdown: 42 }, { markdown: 'x'.repeat(40001) } ]) {
  test(`rejects an unrelated or invalid summary (${Object.keys(changes)[0]})`, async (t) => {
    const { args, comments, warnings } = postCommitFixture();
    const summaryPath = await summaryArtifact(t, JSON.stringify({ runId: args.publishedRunId,
      headSha: args.headSha, markdown: '### Benchmark summary\nStale summary', ...changes }));
    await reportBenchmark({ ...args, summaryPath });
    assert.match(comments[0].body, /summary unavailable/);
    assert.doesNotMatch(comments[0].body, /Stale summary/);
    assert.equal(warnings.length, 1);
  });
}

for (const content of ['not json', 'null']) {
  test(`reports unavailable summary data without suppressing the PR comment (${content})`, async (t) => {
    const { args, comments, warnings } = postCommitFixture();
    const summaryPath = await summaryArtifact(t, content);
    await reportBenchmark({ ...args, summaryPath });
    assert.match(comments[0].body, /summary unavailable/);
    assert.equal(warnings.length, 1);
  });
}

test('a missing artifact does not suppress the PR comment', async () => {
  const { args, comments, warnings } = postCommitFixture();
  await reportBenchmark({ ...args, summaryPath: '/missing/rocjitsu-summary.json' });
  assert.match(comments[0].body, /summary unavailable/);
  assert.equal(warnings.length, 1);
});

test('unpublished runs do not include comparison statistics', async (t) => {
  const { args, comments } = postCommitFixture();
  const summaryPath = await summaryArtifact(t, JSON.stringify({ runId: args.publishedRunId,
    headSha: args.headSha, markdown: '### Benchmark summary\nNot yet published' }));
  await reportBenchmark({ ...args, summaryPath, publishResult: 'failure' });
  assert.doesNotMatch(comments[0].body, /Not yet published/);
});

test('a report-only rerun accepts the original published benchmark summary', async (t) => {
  const { args, comments } = postCommitFixture();
  const summaryPath = await summaryArtifact(t, JSON.stringify({ runId: args.publishedRunId,
    headSha: args.headSha, markdown: '### Benchmark summary\nOriginal benchmark measurements' }));
  await reportBenchmark({ ...args, summaryPath, runAttempt: '3' });
  assert.match(comments[0].body, /Original benchmark measurements/);
});

for (const postMerge of [false, true]) {
  test(`renders the real results summary in the common format (${postMerge ? 'post-merge' : 'presubmit'})`, async (t) => {
    const { resolve } = require('node:path');
    const dashboard = resolve(__dirname, '../../../emulation/rocjitsu/website/dashboard');
    const { args, comments, associated } = postMerge ? postCommitFixture() : fixture();
    args.headSha = '0018f1c70018f1c70018f1c70018f1c70018f1c7';
    args.publishedRunId = 'fictional-develop-23';
    if (postMerge) {
      args.context.sha = args.headSha;
      associated[0].merge_commit_sha = args.headSha;
    }
    const { readFile } = require('node:fs/promises');
    const directory = resolve(dashboard, 'tests/fixtures/data');
    const index = JSON.parse(await readFile(resolve(directory, 'index.json'), 'utf8'));
    const runs = await Promise.all(index.runFiles.map(async (file) => JSON.parse(await readFile(resolve(directory, file), 'utf8'))));
    runs.find((run) => run.publication.id === args.publishedRunId).publication.source.base = {
      branch: 'develop', commit: runs.find((run) => run.publication.id === 'fictional-develop-22').provenance.rocjitsu.rocjitsu_commit_sha,
    };
    const summary = await buildBenchmarkSummary(runs, args.publishedRunId, args.headSha);
    const summaryPath = await summaryArtifact(t, JSON.stringify(summary));
    await reportBenchmark({ ...args, summaryPath });
    const body = comments[0].body;
    assert.match(body, /## rocjitsu benchmarks: success/);
    assert.match(body, /15\/15 benchmarks completed/);
    assert.match(body, /4 faster, 0 slower/);
    assert.match(body, /Fictional GEMM.*35.84 s.*25.4 s.*-29.1%/);
    assert.doesNotMatch(body, /Reporting attempt|Benchmark attempt|Benchmark job|Publication job|shared dashboard dataset/);
  });
}
