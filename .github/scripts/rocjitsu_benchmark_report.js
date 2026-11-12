// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

'use strict';

const { readFile, writeFile } = require('node:fs/promises');
const { resolve } = require('node:path');
const { parseArgs } = require('node:util');
const { findExistingReportComment } = require('./rocjitsu_fuzz_report.js');

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
  publishedRunId,
  headSha,
  summaryPath,
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
  const existing = findExistingReportComment(comments, marker);
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
  let summaryMarkdown = '';
  if (publishResult === 'success' && summaryPath) {
    try {
      const summary = JSON.parse(await readFile(summaryPath, 'utf8'));
      if (summary.runId !== publishedRunId || summary.headSha !== headSha ||
          typeof summary.markdown !== 'string' || summary.markdown.length > 40000) {
        throw new Error('Benchmark summary does not match the published run');
      }
      summaryMarkdown = summary.markdown;
    } catch (error) {
      core?.warning(`Benchmark comparison summary unavailable: ${error.message}`);
    }
  }
  const status = benchmarkResult === 'success' && publishResult !== 'success'
    ? 'results unavailable' : benchmarkResult;
  const phase = postCommit ? 'Post-merge' : 'Presubmit';
  const commitUrl = `${context.serverUrl}/${owner}/${repo}/commit/${headSha}`;
  const body = [
    marker,
    `<!-- rocjitsu-benchmark-report-run:${context.runId}:${runNumber}:${runAttempt} -->`,
    `## rocjitsu benchmarks: ${status}`,
    '',
    `${phase} for commit [\`${headSha.slice(0, 12)}\`](${commitUrl}).`,
    '',
    publishResult === 'success'
      ? summaryMarkdown || 'Result summary unavailable; see the dashboard.'
      : 'Results were not published; see the workflow run.',
    '',
    `[Dashboard](${dashboardUrl.href}) · [View the workflow run](${runUrl}).`,
  ].join('\n');
  // The reporting job serializes the read/compare/write per PR.
  if (existing) {
    await github.rest.issues.updateComment({ owner, repo, comment_id: existing.id, body });
  } else {
    await github.rest.issues.createComment({ owner, repo, issue_number: number, body });
  }
}

const escapeMarkdown = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/@/g, '&#64;').replace(/[\\`*_{}\[\]()!|]/g, '\\$&').replace(/[\r\n]/g, ' ');
const percent = (value) => `${value > 0 ? '+' : ''}${value.toFixed(1)}%`;
const seconds = (value) => `${Number(value.toPrecision(4))} s`;

async function buildBenchmarkSummary(data, runId, headSha) {
  const [{ selectAutomaticReference }, { selectRecentRunSummaries, selectRunComparison },
    { typedJsonIdentity }, { commitShaFor }, { DURATION_CHANGE_BAND_PERCENT }] = await Promise.all([
    import('../../emulation/rocjitsu/website/dashboard/src/data/branchSelectors.js'),
    import('../../emulation/rocjitsu/website/dashboard/src/data/selectors.js'),
    import('../../emulation/rocjitsu/website/dashboard/src/data/jsonValues.js'),
    import('../../emulation/rocjitsu/website/dashboard/src/data/runOrdering.js'),
    import('../../emulation/rocjitsu/website/dashboard/src/utils/performance.js'),
  ]);
  const candidate = data.allRuns.find((run) => run.runId === runId);
  if (!candidate || commitShaFor(candidate) !== headSha) throw new Error('Published candidate does not match measured run and commit');
  // Reruns of the measured revision are not a change baseline.
  const references = { ...data, runs: data.runs.filter((run) => commitShaFor(run) !== headSha) };
  const reference = selectAutomaticReference(references, candidate);
  const baseline = reference.reason === 'exact-base' ? reference.run : null;
  const filters = { targets: data.targets, modes: data.modes, suites: data.suites };
  const [{ completed, failed, timeout, total }] = selectRecentRunSummaries([candidate], filters);
  const failures = [failed && `${failed} failed`, timeout && `${timeout} timed out`].filter(Boolean);
  const lines = [`**${completed}/${total} benchmarks completed.**${failures.length ? ` ${failures.join(', ')}.` : ''}`, ''];
  const summary = { runId, headSha, baselineRunId: baseline?.runId ?? null };
  if (!baseline) {
    lines.push('Timing comparison unavailable: no published result for the recorded base.');
    return { ...summary, markdown: lines.join('\n') };
  }
  const comparisonUrl = new URL('https://rocm.github.io/rocm-systems/rocjitsu-dashboard/');
  comparisonUrl.searchParams.set('view', 'compare');
  comparisonUrl.searchParams.set('compareCandidate', candidate.runId);
  comparisonUrl.searchParams.set('compareBaseline', baseline.runId);
  const environmentChanges = [];
  if (candidate.environmentId !== baseline.environmentId) environmentChanges.push('environment or measurement settings');
  if (candidate.machineId !== baseline.machineId) environmentChanges.push('benchmark machine');
  const tensile = (run) => run.provenance.details.find(({ key }) => key === 'auxiliary.tensile_commit_sha')?.value;
  if (tensile(candidate) !== tensile(baseline)) environmentChanges.push('Tensile revision');
  const workloadSignature = (run) => typedJsonIdentity(run.tests.map((test) =>
    [test.testId, test.suite, test.operation, test.problem]).sort((left, right) => left[0].localeCompare(right[0])));
  if (candidate.catalogId !== baseline.catalogId || workloadSignature(candidate) !== workloadSignature(baseline)) {
    environmentChanges.push('workload definitions or coverage');
  }
  const comparisons = data.targets.flatMap((target) => data.modes.map((mode) => ({
    target, mode, ...selectRunComparison(candidate, baseline, { ...filters, targets: [target], modes: [mode] }),
  }))).filter((comparison) => comparison.comparable.length || comparison.notComparable.length);
  const faster = comparisons.reduce((total, comparison) => total + comparison.counts.faster, 0);
  const slower = comparisons.reduce((total, comparison) => total + comparison.counts.slower, 0);
  lines.push(`**${faster} faster, ${slower} slower** than baseline (±${DURATION_CHANGE_BAND_PERCENT}% duration band).`, '');
  const changes = [];
  for (const comparison of comparisons) {
    const { target, mode, counts } = comparison;
    const unavailable = comparison.notComparable.length + counts.unavailable;
    lines.push(`- ${escapeMarkdown(target)} / ${escapeMarkdown(mode)}: ${counts.faster} faster, ${counts.slower} slower, ${counts.neutral} within band${unavailable ? `, ${unavailable} not comparable` : ''}.`);
    changes.push(...comparison.comparable.filter(({ delta }) => Number.isFinite(delta) && Math.abs(delta) > comparison.tolerance));
  }
  changes.sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta)
    || left.candidateTest.testId.localeCompare(right.candidateTest.testId));
  if (changes.length) {
    lines.push('', 'Largest observed changes (up to 5):', '',
      '| Benchmark | Target / thread policy | Baseline | Current | Duration change |',
      '| --- | --- | ---: | ---: | ---: |');
    for (const { candidateTest, baselineTest, delta } of changes.slice(0, 5)) {
      lines.push(`| ${escapeMarkdown(candidateTest.name)} | ${escapeMarkdown(candidateTest.target)} / ${escapeMarkdown(candidateTest.mode)} | ${seconds(baselineTest.durationSeconds)} | ${seconds(candidateTest.durationSeconds)} | ${percent(delta)} |`);
    }
  } else {
    lines.push('', 'No comparable timing changes outside the duration band.');
  }
  if (environmentChanges.length) lines.push('', `**Changed inputs:** ${environmentChanges.join(', ')}. Timing differences may reflect these changes.`);
  lines.push('', `Compared with [\`${escapeMarkdown(commitShaFor(baseline).slice(0, 12))}\`](${comparisonUrl.href}).`);
  return { ...summary, markdown: lines.join('\n') };
}

async function readBenchmarkSummary(dataDir, runId, headSha) {
  const [{ adaptPublishedDashboardData, validatePublishedManifest }, { parsePublishedJson }] = await Promise.all([
    import('../../emulation/rocjitsu/website/dashboard/src/data/dashboardDataAdapter.js'),
    import('../../emulation/rocjitsu/website/dashboard/src/data/publishedJson.js'),
  ]);
  const index = parsePublishedJson(await readFile(resolve(dataDir, 'index.json'), 'utf8'));
  validatePublishedManifest(index);
  const runs = await Promise.all(index.runFiles.map(async (file) => parsePublishedJson(await readFile(resolve(dataDir, file), 'utf8'))));
  const { data } = adaptPublishedDashboardData({ index, runs });
  return buildBenchmarkSummary(data, runId, headSha);
}

async function summarize() {
  const { values } = parseArgs({ options: {
    'data-dir': { type: 'string' }, 'run-id': { type: 'string' },
    'head-sha': { type: 'string' }, output: { type: 'string' },
  } });
  for (const name of ['data-dir', 'run-id', 'head-sha', 'output']) {
    if (!values[name]) throw new Error(`Missing --${name}`);
  }
  const summary = await readBenchmarkSummary(values['data-dir'], values['run-id'], values['head-sha']);
  await writeFile(values.output, JSON.stringify(summary) + '\n');
}

if (require.main === module) {
  summarize().catch((error) => { console.error(error); process.exitCode = 1; });
}

module.exports = { reportBenchmark, buildBenchmarkSummary, readBenchmarkSummary };
