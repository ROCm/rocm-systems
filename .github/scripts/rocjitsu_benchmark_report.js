// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

'use strict';

const { readFile, writeFile } = require('node:fs/promises');
const { resolve } = require('node:path');
const { parseArgs } = require('node:util');
const { findExistingReportComment } = require('./rocjitsu_fuzz_report.js');

const DASHBOARD_URL = 'https://rocm.github.io/rocm-systems/rocjitsu-dashboard/';
const MAX_SUMMARY_LENGTH = 40000;
const MAX_TIMING_CHANGES = 5;

// The comment job only checks out .github/scripts; data modules are needed by the summary CLI.
function loadBenchmarkData() {
  return import('../../emulation/rocjitsu/scripts/benchmark-data/index.mjs');
}

async function findPullRequest({ github, context, core, headSha }) {
  if (context.eventName !== 'push') return context.payload.pull_request;
  if (context.ref !== 'refs/heads/develop' || headSha !== context.sha) return null;

  const { owner, repo } = context.repo;
  const associated = await github.paginate(
    github.rest.repos.listPullRequestsAssociatedWithCommit,
    { owner, repo, commit_sha: headSha, per_page: 100 },
  );
  const matches = associated.filter((pull) =>
    pull.base?.ref === 'develop' &&
    pull.base?.repo?.full_name === `${owner}/${repo}` &&
    pull.merged_at && pull.merge_commit_sha === headSha,
  );
  if (matches.length !== 1) {
    core?.warning(`Expected one associated develop PR for ${headSha}, found ${matches.length}.`);
    return null;
  }
  return matches[0];
}

function isOlderReport(body, runNumber, runAttempt) {
  const identity = body.match(/<!-- rocjitsu-benchmark-report-run:\d+:(\d+):(\d+) -->/);
  if (!identity) throw new Error('Existing benchmark report has no workflow run identity.');

  const previousRun = BigInt(identity[1]);
  const previousAttempt = BigInt(identity[2]);
  return BigInt(runNumber) < previousRun ||
    (BigInt(runNumber) === previousRun && BigInt(runAttempt) < previousAttempt);
}

async function readSummaryMarkdown({ summaryPath, publishedRunId, headSha, core }) {
  if (!summaryPath) return '';
  try {
    const summary = JSON.parse(await readFile(summaryPath, 'utf8'));
    if (summary.runId !== publishedRunId || summary.headSha !== headSha ||
        typeof summary.markdown !== 'string' || summary.markdown.length > MAX_SUMMARY_LENGTH) {
      throw new Error('Benchmark summary does not match the published run');
    }
    return summary.markdown;
  } catch (error) {
    core?.warning(`Benchmark comparison summary unavailable: ${error.message}`);
    return '';
  }
}

function renderReport({
  context, pull, marker, benchmarkResult, publishResult,
  runNumber, runAttempt, publishedRunId, headSha, summaryMarkdown,
}) {
  const { owner, repo } = context.repo;
  const postMerge = context.eventName === 'push';
  const runUrl = `${context.serverUrl}/${owner}/${repo}/actions/runs/${context.runId}/attempts/${runAttempt}`;
  const commitUrl = `${context.serverUrl}/${owner}/${repo}/commit/${headSha}`;
  const dashboardUrl = new URL(DASHBOARD_URL);
  dashboardUrl.searchParams.set('view', 'branch');
  dashboardUrl.searchParams.set('branch', postMerge ? 'develop' : pull.head.ref);
  dashboardUrl.searchParams.set('detail', '1');

  const published = publishResult === 'success';
  if (published) {
    if (!publishedRunId) throw new Error('Successful publication has no published run ID.');
    dashboardUrl.searchParams.set('run', publishedRunId);
  }
  const status = benchmarkResult === 'success' && !published
    ? 'results unavailable' : benchmarkResult;
  const phase = postMerge ? 'Post-merge' : 'Presubmit';
  const results = published
    ? summaryMarkdown || 'Result summary unavailable; see the dashboard.'
    : 'Results were not published; see the workflow run.';

  return [
    marker,
    `<!-- rocjitsu-benchmark-report-run:${context.runId}:${runNumber}:${runAttempt} -->`,
    `## rocjitsu benchmarks: ${status}`,
    '',
    `${phase} for commit [\`${headSha.slice(0, 12)}\`](${commitUrl}).`,
    '',
    results,
    '',
    `[Dashboard](${dashboardUrl.href}) · [View the workflow run](${runUrl}).`,
  ].join('\n');
}

async function reportBenchmark(options) {
  const { github, context, runNumber, runAttempt, headSha } = options;
  const pull = await findPullRequest(options);
  if (!pull) return;

  const marker = context.eventName === 'push'
    ? `<!-- rocjitsu-post-commit-benchmark:${headSha} -->`
    : `<!-- rocjitsu-benchmark-report:${pull.number} -->`;
  const target = { ...context.repo, issue_number: pull.number };
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...target, per_page: 100,
  });
  const existing = findExistingReportComment(comments, marker);
  if (existing && isOlderReport(existing.body, runNumber, runAttempt)) return;

  const summaryMarkdown = options.publishResult === 'success'
    ? await readSummaryMarkdown(options) : '';
  const body = renderReport({ ...options, pull, marker, summaryMarkdown });

  // The reporting job serializes comment lookup and mutation per PR.
  if (existing) {
    await github.rest.issues.updateComment({
      ...context.repo, comment_id: existing.id, body,
    });
  } else {
    await github.rest.issues.createComment({ ...target, body });
  }
}

function escapeMarkdown(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/@/g, '&#64;')
    .replace(/[\\`*_{}[\]()!|]/g, '\\$&')
    .replace(/[\r\n]/g, ' ');
}

function formatPercent(value) {
  return `${value > 0 ? '+' : ''}${value.toFixed(1)}%`;
}

function formatSeconds(value) {
  return `${Number(value.toPrecision(4))} s`;
}

function describeInputChanges(candidate, baseline, typedJsonIdentity) {
  const changes = [];
  if (candidate.environmentId !== baseline.environmentId) {
    changes.push('environment or measurement settings');
  }
  if (candidate.machineId !== baseline.machineId) changes.push('benchmark machine');

  const tensileRevision = (run) => run.provenance.details
    .find(({ key }) => key === 'auxiliary.tensile_commit_sha')?.value;
  if (tensileRevision(candidate) !== tensileRevision(baseline)) changes.push('Tensile revision');

  const workloadSignature = (run) => typedJsonIdentity(
    run.tests.map((test) => [test.testId, test.suite, test.operation, test.problem])
      .sort((left, right) => left[0].localeCompare(right[0])),
  );
  if (candidate.catalogId !== baseline.catalogId ||
      workloadSignature(candidate) !== workloadSignature(baseline)) {
    changes.push('workload definitions or coverage');
  }
  return changes;
}

function renderCoverage({ completed, total, failed, timeout }) {
  const failures = [];
  if (failed) failures.push(`${failed} failed`);
  if (timeout) failures.push(`${timeout} timed out`);
  const counts = `**${completed}/${total} benchmarks completed.**`;
  return failures.length ? `${counts} ${failures.join(', ')}.` : counts;
}

function renderTimingChanges(comparisons) {
  const changes = comparisons.flatMap((comparison) => comparison.comparable.filter(
    ({ delta }) => Number.isFinite(delta) && Math.abs(delta) > comparison.tolerance,
  ));
  changes.sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta) ||
    left.candidateTest.testId.localeCompare(right.candidateTest.testId));
  if (!changes.length) return ['No comparable timing changes outside the duration band.'];

  const lines = [
    `Largest observed changes (up to ${MAX_TIMING_CHANGES}):`,
    '',
    '| Benchmark | Target / thread policy | Baseline | Current | Duration change |',
    '| --- | --- | ---: | ---: | ---: |',
  ];
  for (const { candidateTest, baselineTest, delta } of changes.slice(0, MAX_TIMING_CHANGES)) {
    const benchmark = escapeMarkdown(candidateTest.name);
    const configuration = `${escapeMarkdown(candidateTest.target)} / ${escapeMarkdown(candidateTest.mode)}`;
    const durations = `${formatSeconds(baselineTest.durationSeconds)} | ${formatSeconds(candidateTest.durationSeconds)}`;
    lines.push(`| ${benchmark} | ${configuration} | ${durations} | ${formatPercent(delta)} |`);
  }
  return lines;
}

function renderComparison(comparisons, band) {
  const faster = comparisons.reduce((total, comparison) => total + comparison.counts.faster, 0);
  const slower = comparisons.reduce((total, comparison) => total + comparison.counts.slower, 0);
  const lines = [
    `**${faster} faster, ${slower} slower** than baseline (±${band}% duration band).`,
    '',
  ];
  for (const { target, mode, counts, notComparable } of comparisons) {
    const unavailable = notComparable.length + counts.unavailable;
    const configuration = `${escapeMarkdown(target)} / ${escapeMarkdown(mode)}`;
    const countsText = `${counts.faster} faster, ${counts.slower} slower, ${counts.neutral} within band`;
    const missing = unavailable ? `, ${unavailable} not comparable` : '';
    lines.push(`- ${configuration}: ${countsText}${missing}.`);
  }
  lines.push('', ...renderTimingChanges(comparisons));
  return lines;
}

async function buildBenchmarkSummary(data, runId, headSha) {
  const {
    findRecordedBase, selectRecentRunSummaries, selectRunComparison,
    typedJsonIdentity, commitShaFor, DURATION_CHANGE_BAND_PERCENT,
  } = await loadBenchmarkData();
  const candidate = data.allRuns.find((run) => run.runId === runId);
  if (!candidate || commitShaFor(candidate) !== headSha) {
    throw new Error('Published candidate does not match measured run and commit');
  }
  const baseline = findRecordedBase(data.runs, candidate, data.canonicalBranch);
  const filters = { targets: data.targets, modes: data.modes, suites: data.suites };
  const [coverage] = selectRecentRunSummaries([candidate], filters);
  const lines = [renderCoverage(coverage), ''];
  const summary = { runId, headSha, baselineRunId: baseline?.runId ?? null };
  if (!baseline) {
    lines.push('Timing comparison unavailable: no published result for the recorded base.');
    return { ...summary, markdown: lines.join('\n') };
  }

  const comparisons = [];
  for (const target of data.targets) {
    for (const mode of data.modes) {
      const comparison = selectRunComparison(candidate, baseline, {
        ...filters, targets: [target], modes: [mode],
      });
      if (comparison.comparable.length || comparison.notComparable.length) {
        comparisons.push({ target, mode, ...comparison });
      }
    }
  }
  lines.push(...renderComparison(comparisons, DURATION_CHANGE_BAND_PERCENT));
  const inputChanges = describeInputChanges(candidate, baseline, typedJsonIdentity);
  if (inputChanges.length) {
    lines.push('', `**Changed inputs:** ${inputChanges.join(', ')}. Timing differences may reflect these changes.`);
  }

  const comparisonUrl = new URL(DASHBOARD_URL);
  comparisonUrl.searchParams.set('view', 'compare');
  comparisonUrl.searchParams.set('compareCandidate', candidate.runId);
  comparisonUrl.searchParams.set('compareBaseline', baseline.runId);
  const baselineCommit = escapeMarkdown(commitShaFor(baseline).slice(0, 12));
  lines.push('', `Compared with [\`${baselineCommit}\`](${comparisonUrl.href}).`);
  return { ...summary, markdown: lines.join('\n') };
}

async function readBenchmarkSummary(dataDir, runId, headSha) {
  const { normalizePublishedData, validatePublishedManifest, parsePublishedJson } = await loadBenchmarkData();
  const index = parsePublishedJson(await readFile(resolve(dataDir, 'index.json'), 'utf8'));
  validatePublishedManifest(index);
  const runs = await Promise.all(index.runFiles.map(async (file) => {
    const text = await readFile(resolve(dataDir, file), 'utf8');
    return parsePublishedJson(text);
  }));
  const data = normalizePublishedData({ index, runs });
  return buildBenchmarkSummary(data, runId, headSha);
}

async function runSummaryCommand() {
  const { values } = parseArgs({ options: {
    'data-dir': { type: 'string' },
    'run-id': { type: 'string' },
    'head-sha': { type: 'string' },
    output: { type: 'string' },
  } });
  for (const name of ['data-dir', 'run-id', 'head-sha', 'output']) {
    if (!values[name]) throw new Error(`Missing --${name}`);
  }
  const summary = await readBenchmarkSummary(
    values['data-dir'], values['run-id'], values['head-sha'],
  );
  await writeFile(values.output, JSON.stringify(summary) + '\n');
}

if (require.main === module) {
  runSummaryCommand().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

module.exports = { reportBenchmark, buildBenchmarkSummary, readBenchmarkSummary };
