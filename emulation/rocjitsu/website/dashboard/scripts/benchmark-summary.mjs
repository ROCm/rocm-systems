// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { adaptPublishedDashboardData, validatePublishedManifest } from '../src/data/dashboardDataAdapter.js';
import { parsePublishedJson } from '../src/data/publishedJson.js';
import { selectAutomaticReference } from '../src/data/branchSelectors.js';
import { selectRecentRunSummaries, selectRunComparison } from '../src/data/selectors.js';
import { typedJsonIdentity } from '../src/data/jsonValues.js';
import { commitShaFor } from '../src/data/runOrdering.js';
import { DURATION_CHANGE_BAND_PERCENT } from '../src/utils/performance.js';

const escapeMarkdown = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/@/g, '&#64;').replace(/[\\`*_{}\[\]()!|]/g, '\\$&').replace(/[\r\n]/g, ' ');
const percent = (value) => `${value > 0 ? '+' : ''}${value.toFixed(1)}%`;
const seconds = (value) => `${Number(value.toPrecision(4))} s`;

export function buildBenchmarkSummary(data, runId, headSha) {
  const candidate = data.allRuns.find((run) => run.runId === runId);
  if (!candidate || commitShaFor(candidate) !== headSha) throw new Error('Published candidate does not match measured run and commit');
  // Reruns of the measured revision are not a change baseline.
  const references = { ...data, runs: data.runs.filter((run) => commitShaFor(run) !== headSha) };
  const { run: baseline, reason } = selectAutomaticReference(references, candidate);
  const filters = { targets: data.targets, modes: data.modes, suites: data.suites };
  const [{ completed, failed, timeout, total }] = selectRecentRunSummaries([candidate], filters);
  const lines = ['### Benchmark summary', '',
    `Coverage: **${completed}/${total} completed**, **${failed} failed**, **${timeout} timed out**.`, ''];
  const summary = { runId, headSha, baselineRunId: baseline?.runId ?? null };
  if (!baseline) {
    lines.push('No published develop baseline is available; timing changes are unavailable.');
    return { ...summary, markdown: lines.join('\n') };
  }
  const comparisonUrl = new URL('https://rocm.github.io/rocm-systems/rocjitsu-dashboard/');
  comparisonUrl.searchParams.set('view', 'compare');
  comparisonUrl.searchParams.set('compareCandidate', candidate.runId);
  comparisonUrl.searchParams.set('compareBaseline', baseline.runId);
  lines.push(`Baseline: commit \`${escapeMarkdown(commitShaFor(baseline))}\`, run \`${escapeMarkdown(baseline.runId)}\`.`, '',
    reason === 'exact-base'
      ? 'Baseline matches the recorded develop base (the first parent for post-merge runs).'
      : '**No result for the recorded base.** Using the latest develop execution completed before this run; the difference may include other changes.', '');

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
  if (environmentChanges.length) lines.push(`**Changed inputs:** ${environmentChanges.join(', ')}. Timing differences may reflect these changes.`, '');

  lines.push(`Observed duration changes use the dashboard's ±${DURATION_CHANGE_BAND_PERCENT}% band; this is not a statistical significance test. Lower duration is faster.`, '',
    '| Target / thread policy | Faster | Slower | Within band | Not comparable |',
    '| --- | ---: | ---: | ---: | ---: |');
  const changes = [];
  for (const target of data.targets) {
    for (const mode of data.modes) {
      const comparison = selectRunComparison(candidate, baseline, { ...filters, targets: [target], modes: [mode] });
      if (!comparison.comparable.length && !comparison.notComparable.length) continue;
      const { faster, slower, neutral } = comparison.counts;
      lines.push(`| ${escapeMarkdown(target)} / ${escapeMarkdown(mode)} | ${faster} | ${slower} | ${neutral} | ${comparison.notComparable.length + comparison.counts.unavailable} |`);
      changes.push(...comparison.comparable.filter(({ delta }) => Number.isFinite(delta) && Math.abs(delta) > comparison.tolerance));
    }
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
    lines.push('', 'No comparable timing changes outside the dashboard band.');
  }
  lines.push('', '[Compare these exact runs on the dashboard](' + comparisonUrl.href + ').');
  return { ...summary, markdown: lines.join('\n') };
}

export async function readBenchmarkSummary(dataDir, runId, headSha) {
  const index = parsePublishedJson(await readFile(resolve(dataDir, 'index.json'), 'utf8'));
  validatePublishedManifest(index);
  const runs = await Promise.all(index.runFiles.map(async (file) => parsePublishedJson(await readFile(resolve(dataDir, file), 'utf8'))));
  const { data } = adaptPublishedDashboardData({ index, runs });
  return buildBenchmarkSummary(data, runId, headSha);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
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
