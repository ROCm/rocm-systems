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
const DURATION_CHANGE_BAND_PERCENT = 3;
const THREAD_POLICIES = ['single', 'default'];

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
  runNumber, runAttempt, benchmarkRunAttempt, publishedRunId, headSha, summaryMarkdown,
}) {
  const { owner, repo } = context.repo;
  const postMerge = context.eventName === 'push';
  const workflowAttempt = benchmarkRunAttempt || runAttempt;
  const runUrl = `${context.serverUrl}/${owner}/${repo}/actions/runs/${context.runId}/attempts/${workflowAttempt}`;
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
  const phase = postMerge ? 'Post-merge' : 'Presubmit';
  const results = published
    ? summaryMarkdown || 'Result summary unavailable; see the dashboard.'
    : 'Results were not published; see the workflow run.';

  return [
    marker,
    `<!-- rocjitsu-benchmark-report-run:${context.runId}:${runNumber}:${runAttempt} -->`,
    `## rocjitsu benchmark job: ${benchmarkResult}`,
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

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${stableJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function describeInputChanges(candidate, baseline) {
  const changes = [];
  const measurementInputs = (run) => {
    const provenance = structuredClone(run.provenance ?? {});
    for (const key of ['rocjitsu_commit_sha', 'rocjitsu_commit_timestamp', 'target_config_sha256']) {
      delete provenance.rocjitsu?.[key];
    }
    delete provenance.machine?.hostname;
    const settings = ['warmups', 'samples', 'timeout_seconds', 'benchmark_suite']
      .map((key) => [key, run.execution_summary[key]]);
    return stableJson({ provenance, settings });
  };
  if (measurementInputs(candidate) !== measurementInputs(baseline)) {
    changes.push('environment or measurement settings');
  }
  if (candidate.publication.machine_label !== baseline.publication.machine_label) {
    changes.push('benchmark machine');
  }
  if (candidate.publication.tensile_commit_sha !== baseline.publication.tensile_commit_sha) {
    changes.push('Tensile revision');
  }
  const workloadSignature = (run) => stableJson(run.benchmark_results.map((result) => [
    result.target, result.thread_policy, result.id, result.suite, result.operation, result.problem,
  ]).sort((left, right) => stableJson(left).localeCompare(stableJson(right))));
  if (workloadSignature(candidate) !== workloadSignature(baseline)) {
    changes.push('workload definitions or coverage');
  }
  return changes;
}

function medianSeconds(result) {
  const samples = result.timing_results_s;
  if (result.status !== 'completed' || !Array.isArray(samples) || !samples.length ||
      !samples.every((sample) => Number.isFinite(sample) && sample > 0)) return null;

  const sorted = [...samples].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : sorted[middle - 1] / 2 + sorted[middle] / 2;
}

function indexResults(run) {
  const results = new Map();
  for (const result of run.benchmark_results) {
    if (typeof result.id !== 'string' || !result.id || typeof result.name !== 'string' ||
        !result.name.trim() || typeof result.target !== 'string' ||
        !/^[A-Za-z0-9._-]+$/.test(result.target) || !THREAD_POLICIES.includes(result.thread_policy)) {
      throw new Error('Invalid benchmark result identity');
    }
    const testId = `${result.target}/${result.thread_policy}:${result.id}`;
    if (results.has(testId)) throw new Error(`Duplicate benchmark result: ${testId}`);
    results.set(testId, {
      ...result, testId, mode: result.thread_policy, durationSeconds: medianSeconds(result),
    });
  }
  return results;
}

function compareBenchmarkResults(candidateResults, baselineResults) {
  const configurations = new Map();
  const ids = new Set([...candidateResults.keys(), ...baselineResults.keys()]);
  for (const id of ids) {
    const candidateTest = candidateResults.get(id);
    const baselineTest = baselineResults.get(id);
    const { target, mode } = candidateTest ?? baselineTest;
    const key = `${target}/${mode}`;
    if (!configurations.has(key)) {
      configurations.set(key, {
        target, mode, comparable: [], notComparable: [],
        counts: { faster: 0, slower: 0, neutral: 0, unavailable: 0 },
      });
    }
    const comparison = configurations.get(key);
    if (candidateTest?.durationSeconds == null || baselineTest?.durationSeconds == null) {
      comparison.notComparable.push({ candidateTest, baselineTest });
      continue;
    }
    const delta = (candidateTest.durationSeconds - baselineTest.durationSeconds) /
      baselineTest.durationSeconds * 100;
    if (!Number.isFinite(delta)) {
      comparison.counts.unavailable += 1;
      continue;
    }
    // Subtraction and division can put an exact boundary a few ulps outside the band.
    const roundingTolerance = Number.EPSILON * Math.max(100, Math.abs(delta)) * 4;
    const outsideBand = Math.abs(delta) > DURATION_CHANGE_BAND_PERCENT + roundingTolerance;
    const classification = outsideBand ? (delta > 0 ? 'slower' : 'faster') : 'neutral';
    comparison.comparable.push({ candidateTest, baselineTest, delta, classification });
    comparison.counts[classification] += 1;
  }
  return [...configurations.values()].sort((left, right) =>
    left.target.localeCompare(right.target) ||
    THREAD_POLICIES.indexOf(left.mode) - THREAD_POLICIES.indexOf(right.mode));
}

function findRecordedBase(runs, candidate) {
  const base = candidate.publication.source.base;
  if (base?.branch !== 'develop') return null;
  return runs.filter((run) => run.publication.source.branch === 'develop' &&
    run.provenance.rocjitsu.rocjitsu_commit_sha === base.commit &&
    run.provenance.rocjitsu.rocjitsu_commit_sha !== candidate.provenance.rocjitsu.rocjitsu_commit_sha)
    .sort((left, right) => Date.parse(left.execution_summary.finished_at) -
      Date.parse(right.execution_summary.finished_at) ||
      left.publication.id.localeCompare(right.publication.id)).at(-1) ?? null;
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
    ({ classification }) => classification !== 'neutral',
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

// Select a weekly reference for the lightweight published-run summary.
// The standalone bot's calculations can differ from the dashboard.
function overviewReferences(runs, candidate, candidateResults) {
  const time = (value) => typeof value === 'string' ? Date.parse(value) : NaN;
  const cutoff = time(candidate.execution_summary.timestamp);
  const available = runs.filter((run) => run !== candidate &&
    time(run.execution_summary.finished_at) < cutoff);
  const byExecution = (left, right) => time(left.execution_summary.finished_at) -
    time(right.execution_summary.finished_at) || left.publication.id.localeCompare(right.publication.id);
  const develop = available.filter((run) => run.publication.source.branch === 'develop');
  const latest = [...develop].sort(byExecution).at(-1) ?? null;
  const sha = (run) => run.provenance.rocjitsu.rocjitsu_commit_sha;
  const commitTime = (run) => time(run.provenance.rocjitsu.rocjitsu_commit_timestamp);
  const ordered = develop.filter((run) => Number.isFinite(commitTime(run))).sort((left, right) =>
    (sha(left) === sha(right) ? 0 : commitTime(left) - commitTime(right) || sha(left).localeCompare(sha(right))) ||
    byExecution(left, right));
  if (!ordered.length) return { latest, weekly: null };
  const day = (run) => Math.floor(commitTime(run) / 86400000);
  const anchor = day(ordered.at(-1));
  const weekly = ordered.filter((run) => day(run) >= anchor - 6 && day(run) <= anchor);

  // Published catalog membership is immutable per configuration. Like the adapter,
  // take its first definition before computing the range-wide cohort or eligibility.
  const catalogs = new Map();
  for (const run of available) {
    const id = run.execution_summary.benchmark_suite;
    if (!catalogs.has(id)) catalogs.set(id, new Map());
    const catalog = catalogs.get(id);
    const configurations = new Map();
    for (const result of run.benchmark_results) {
      const key = `${result.target}/${result.thread_policy}`;
      if (!configurations.has(key)) configurations.set(key, []);
      configurations.get(key).push(result);
    }
    for (const [key, results] of configurations) {
      if (!catalog.has(key)) catalog.set(key, results);
    }
  }
  const selected = [...candidateResults.values()];
  const targets = [...new Set(selected.map((result) => result.target))].sort();
  const modes = THREAD_POLICIES.filter((mode) => selected.some((result) => result.mode === mode));
  const suites = new Set(selected.map((result) => result.suite));
  const cohorts = targets.flatMap((target) => modes.map((mode) => {
    const key = `${target}/${mode}`;
    const memberships = [...new Set(weekly.filter((run) => run.benchmark_results.some(
      (result) => result.target === target && result.thread_policy === mode,
    )).map((run) => run.execution_summary.benchmark_suite))]
      .map((id) => catalogs.get(id)?.get(key) ?? []);
    return (memberships[0] ?? []).filter((result) => suites.has(result.suite) &&
      memberships.every((rows) => rows.some((row) => row.id === result.id)))
      .map((result) => `${key}:${result.id}`);
  }));
  const reference = weekly.find((run) => {
    const results = indexResults(run);
    return cohorts.length > 0 && cohorts.every((ids) => {
      const durations = ids.map((id) => results.get(id)?.durationSeconds);
      const total = durations.reduce((sum, duration) => sum + duration, 0);
      return ids.length > 0 && durations.every((duration) => duration != null) &&
        Number.isFinite(total) && total > 0;
    });
  }) ?? null;
  return { latest, weekly: reference };
}

function renderPerformanceOverview(runs, candidate, candidateResults) {
  const { latest, weekly } = overviewReferences(runs, candidate, candidateResults);
  const commit = (run) => {
    const sha = run?.provenance.rocjitsu.rocjitsu_commit_sha;
    return typeof sha === 'string' && sha ?
      `[\`${escapeMarkdown(sha.slice(0, 12))}\`](https://github.com/ROCm/rocm-systems/commit/${encodeURIComponent(sha)})` : 'N/A';
  };
  const sources = [candidate, latest, weekly];
  const indexes = sources.map((run) => run ? indexResults(run) : null);
  const configurations = [...new Set([...candidateResults.values()].map(
    (result) => `${result.target}/${result.mode}`,
  ))].sort((left, right) => left.split('/')[0].localeCompare(right.split('/')[0]) ||
    THREAD_POLICIES.indexOf(left.split('/')[1]) - THREAD_POLICIES.indexOf(right.split('/')[1]));
  const lines = [
    '### Performance overview', '',
    `- **Candidate commit:** ${commit(candidate)}`,
    `- **Latest develop reference:** ${commit(latest)}`,
    `- **1W reference:** ${commit(weekly)}`, '',
    '| Target | Thread policy | Matched benchmarks | Candidate duration | Latest develop duration / speedup | 1W reference duration / speedup |',
    '| --- | --- | ---: | ---: | ---: | ---: |',
  ];
  for (const configuration of configurations) {
    const [target, mode] = configuration.split('/');
    const selected = indexes.filter(Boolean).map((index) => new Map([...index].filter(
      ([, result]) => result.target === target && result.mode === mode,
    )));
    const allIds = new Set(selected.flatMap((index) => [...index.keys()]));
    const sharedIds = [...allIds].filter((id) => selected.every((index) => index.has(id)));
    const durations = indexes.map((index) => {
      if (!index || !sharedIds.length) return null;
      const values = sharedIds.map((id) => index.get(id)?.durationSeconds);
      const total = values.reduce((sum, value) => sum + value, 0);
      return values.every((value) => value != null) && Number.isFinite(total) && total > 0
        ? Number(total.toPrecision(15)) : null;
    });
    const [current, ...references] = durations;
    const duration = (value) => value === null ? 'N/A' : formatSeconds(value);
    const comparison = (value) => {
      const speedup = value !== null && current !== null ? value / current : NaN;
      if (value === null) return 'N/A';
      return `${duration(value)} / ${Number.isFinite(speedup) ? `${speedup.toFixed(3)}×` : 'N/A'}`;
    };
    lines.push(`| ${escapeMarkdown(target)} | ${escapeMarkdown(mode)} | ${sharedIds.length}/${allIds.size} | ` +
      `${duration(current)} | ${references.map(comparison).join(' | ')} |`);
  }
  if (!configurations.length) lines.push('| N/A | N/A | 0/0 | N/A | N/A | N/A |');
  lines.push('', 'Durations sum benchmark medians over the same matched set in each row (matched/union across available runs). ' +
    'Failed or invalid matched results make that duration N/A; speedup is reference / candidate. ' +
    'References use only measurements completed before the candidate measurement started.');
  return { lines, latestDevelopRunId: latest?.publication.id ?? null, weeklyRunId: weekly?.publication.id ?? null };
}

function buildBenchmarkSummary(runs, runId, headSha) {
  const candidate = runs.find((run) => run.publication.id === runId);
  if (!candidate || candidate.provenance.rocjitsu.rocjitsu_commit_sha !== headSha) {
    throw new Error('Published candidate does not match measured run and commit');
  }
  const candidateResults = indexResults(candidate);
  const coverage = { total: candidateResults.size, completed: 0, failed: 0, timeout: 0 };
  for (const result of candidateResults.values()) {
    if (result.durationSeconds !== null) coverage.completed += 1;
    else if (result.status === 'failed') coverage.failed += 1;
    else if (result.status === 'timeout') coverage.timeout += 1;
  }
  const baseline = findRecordedBase(runs, candidate);
  const overview = renderPerformanceOverview(runs, candidate, candidateResults);
  const lines = [...overview.lines, '', '### Recorded-base comparison', '', renderCoverage(coverage), ''];
  const summary = { runId, headSha, baselineRunId: baseline?.publication.id ?? null,
    overview: { latestDevelopRunId: overview.latestDevelopRunId, weeklyRunId: overview.weeklyRunId } };
  if (!baseline) {
    lines.push('Timing comparison unavailable: no published result for the recorded base.', '',
      'The Dashboard link chooses its own reference and may compare against an earlier develop run.');
    return { ...summary, markdown: lines.join('\n') };
  }

  const baselineResults = indexResults(baseline);
  const comparisons = compareBenchmarkResults(candidateResults, baselineResults);
  const hasComparableTimings = comparisons.some((comparison) => comparison.comparable.length);
  const sampling = (run) => {
    const { samples = 'unknown', warmups = 'unknown' } = run.execution_summary;
    return `${escapeMarkdown(samples)} sample${samples === 1 ? '' : 's'} and ` +
      `${escapeMarkdown(warmups)} warmup${warmups === 1 ? '' : 's'}`;
  };
  lines.push(`Sampling per benchmark: current ${sampling(candidate)}; baseline ${sampling(baseline)}.`, '');
  if (hasComparableTimings) lines.push(...renderComparison(comparisons, DURATION_CHANGE_BAND_PERCENT));
  else lines.push('Timing comparison unavailable: no comparable benchmark timings.');
  const inputChanges = describeInputChanges(candidate, baseline);
  if (inputChanges.length) {
    const caveat = hasComparableTimings ? ' Timing differences may reflect these changes.' : '';
    lines.push('', `**Changed inputs:** ${inputChanges.join(', ')}.${caveat}`);
  }

  const comparisonUrl = new URL(DASHBOARD_URL);
  comparisonUrl.searchParams.set('view', 'compare');
  comparisonUrl.searchParams.set('compareCandidate', candidate.publication.id);
  comparisonUrl.searchParams.set('compareBaseline', baseline.publication.id);
  const results = [...candidateResults.values(), ...baselineResults.values()];
  for (const [parameter, field] of [
    ['compareTargets', 'target'], ['compareSuites', 'suite'], ['compareModes', 'mode'],
  ]) {
    const values = [...new Set(results.map((result) => result[field]))].sort();
    if (!values.length) comparisonUrl.searchParams.set(parameter, '');
    for (const value of values) comparisonUrl.searchParams.append(parameter, value);
  }
  const baselineCommit = escapeMarkdown(baseline.provenance.rocjitsu.rocjitsu_commit_sha.slice(0, 12));
  lines.push('', `Compared with [\`${baselineCommit}\`](${comparisonUrl.href}).`);
  return { ...summary, markdown: lines.join('\n') };
}

async function readBenchmarkSummary(dataDir, runId, headSha) {
  const index = JSON.parse(await readFile(resolve(dataDir, 'index.json'), 'utf8'));
  const validPath = /^runs\/(default-branch|side-branches)\/[A-Za-z0-9][A-Za-z0-9._-]*\.json$/;
  if (!Array.isArray(index.runFiles) || new Set(index.runFiles).size !== index.runFiles.length) {
    throw new Error('Invalid publication index');
  }
  for (const file of index.runFiles) {
    if (typeof file !== 'string' || !validPath.test(file)) {
      throw new Error(`Invalid run filename: ${file}`);
    }
  }
  const runs = await Promise.all(index.runFiles.map(async (file) => {
    const run = JSON.parse(await readFile(resolve(dataDir, file), 'utf8'));
    const branch = run.publication?.source?.branch;
    const group = branch === 'develop' ? 'default-branch' : 'side-branches';
    if (run.schema_version !== 1 || typeof branch !== 'string' || !branch ||
        !Array.isArray(run.benchmark_results) || !run.provenance?.rocjitsu ||
        !run.execution_summary || file !== `runs/${group}/${run.publication.id}.json`) {
      throw new Error(`Invalid published run: ${file}`);
    }
    return run;
  }));
  if (new Set(runs.map((run) => run.publication.id)).size !== runs.length) {
    throw new Error('Duplicate published run ID');
  }
  return buildBenchmarkSummary(runs, runId, headSha);
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
