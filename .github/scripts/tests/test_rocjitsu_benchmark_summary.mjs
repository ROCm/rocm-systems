// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { adaptPublishedDashboardData } from '../../../emulation/rocjitsu/website/dashboard/src/data/dashboardDataAdapter.js';
import { publishedRunPath } from '../../../emulation/rocjitsu/website/dashboard/src/data/runSchema.js';
import { buildBenchmarkSummary, readBenchmarkSummary } from '../../../emulation/rocjitsu/website/dashboard/scripts/benchmark-summary.mjs';

const fixture = JSON.parse(await readFile(new URL('../../../emulation/rocjitsu/website/dashboard/tests/fixtures/fictional-producer-run.json', import.meta.url)));
const candidateSha = 'c'.repeat(40);
const baselineSha = 'b'.repeat(40);

function run(id, sha, timestamp) {
  const result = structuredClone(fixture);
  result.publication.id = id;
  result.publication.source = { branch: 'develop', base: null };
  result.provenance.rocjitsu.rocjitsu_commit_sha = sha;
  result.execution_summary.finished_at = timestamp;
  for (const row of result.benchmark_results) row.timing_results_s = [100];
  return result;
}

function publication() {
  const baseline = run('baseline', baselineSha, '2026-11-01T00:00:00Z');
  const candidate = run('candidate', candidateSha, '2026-11-02T00:00:00Z');
  candidate.publication.source.base = { branch: 'develop', commit: baselineSha };
  candidate.benchmark_results[0].timing_results_s = [120];
  candidate.benchmark_results[1].timing_results_s = [80];
  return [baseline, candidate];
}

function dataset(runs) {
  return { index: { generatedAt: '2026-11-03T00:00:00Z', runFiles: runs.map((entry) => publishedRunPath(entry)) }, runs };
}

function summarize(runs = publication()) {
  return buildBenchmarkSummary(adaptPublishedDashboardData(dataset(runs)).data, 'candidate', candidateSha);
}

test('uses dashboard comparisons for counts, timing rows and exact run links', () => {
  const summary = summarize();
  assert.equal(summary.baselineRunId, 'baseline');
  assert.match(summary.markdown, /3\/3 completed/);
  assert.match(summary.markdown, /\| gfx950 \/ default \| 1 \| 1 \| 1 \| 0 \|/);
  assert.match(summary.markdown, /\| 100 s \| 120 s \| \+20.0% \|/);
  assert.match(summary.markdown, /\| 100 s \| 80 s \| -20.0% \|/);
  assert.match(summary.markdown, /first parent/);
  assert.match(summary.markdown, /not a statistical significance test/);
  const url = new URL(summary.markdown.match(/\[Compare these exact runs on the dashboard\]\(([^)]+)\)/)[1]);
  assert.equal(url.searchParams.get('compareBaseline'), 'baseline');
  assert.equal(url.searchParams.get('compareCandidate'), 'candidate');
  assert.equal(url.searchParams.get('view'), 'compare');
});

test('prefers the recorded parent over a more recent unrelated develop result', () => {
  const runs = publication();
  runs.push(run('other', 'd'.repeat(40), '2026-11-01T12:00:00Z'));
  assert.equal(summarize(runs).baselineRunId, 'baseline');
});

test('fallback identifies that the difference may include other changes and skips same-commit reruns', () => {
  const runs = publication();
  runs[1].publication.source.base.commit = 'd'.repeat(40);
  runs.push(run('previous-attempt', candidateSha, '2026-11-01T12:00:00Z'));
  const summary = summarize(runs);
  assert.equal(summary.baselineRunId, 'baseline');
  assert.match(summary.markdown, /No result for the recorded base/);
  assert.match(summary.markdown, /may include other changes/);
});

test('does not invent a comparison when no baseline exists', () => {
  const summary = summarize(publication().slice(1));
  assert.equal(summary.baselineRunId, null);
  assert.match(summary.markdown, /3\/3 completed/);
  assert.match(summary.markdown, /No published develop baseline/);
  assert.doesNotMatch(summary.markdown, /Duration change \|/);
});

test('separates failed and timed-out results from comparable timing changes', () => {
  const runs = publication();
  runs[1].benchmark_results[0].status = 'failed';
  runs[1].benchmark_results[1].status = 'timeout';
  const { markdown } = summarize(runs);
  assert.match(markdown, /1\/3 completed.*1 failed.*1 timed out/);
  assert.match(markdown, /\| gfx950 \/ default \| 0 \| 0 \| 1 \| 2 \|/);
  assert.doesNotMatch(markdown, /\+20.0%/);
});

test('flags changed environment, machine, dependencies and workload inputs', () => {
  const runs = publication();
  runs[1].provenance.machine.cpu = 'other-cpu';
  runs[1].publication.machine_label = 'other-host';
  runs[1].publication.tensile_commit_sha = 'd'.repeat(40);
  runs[1].benchmark_results[0].problem.rows += 1;
  const { markdown } = summarize(runs);
  assert.match(markdown, /Changed inputs:.*environment.*machine.*Tensile.*workload/);
  assert.match(markdown, /Timing differences may reflect these changes/);
});

test('escapes untrusted benchmark names in Markdown tables', () => {
  const runs = publication();
  runs[1].benchmark_results[0].name = '<img> | [link](url)\n@someone';
  const { markdown } = summarize(runs);
  assert.ok(markdown.includes('&lt;img&gt; \\| \\[link\\]\\(url\\) &#64;someone'));
  assert.ok(!markdown.includes('<img>'));
});

test('uses medians from the dashboard adapter instead of the first sample', () => {
  const runs = publication();
  runs[1].benchmark_results[0].timing_results_s = [1, 120, 130];
  assert.match(summarize(runs).markdown, /\| 100 s \| 120 s \| \+20.0% \|/);
});

test('groups targets and thread policies separately and limits the largest changes', () => {
  const runs = publication();
  for (const entry of runs) {
    const extra = entry.benchmark_results.map((row) => ({ ...structuredClone(row), target: 'gfx1250', thread_policy: 'single' }));
    entry.benchmark_results.push(...extra);
    if (entry.publication.id === 'candidate') {
      for (const row of entry.benchmark_results) row.timing_results_s = [150];
    }
  }
  const { markdown } = summarize(runs);
  assert.match(markdown, /\| gfx950 \/ default \| 0 \| 3 \| 0 \| 0 \|/);
  assert.match(markdown, /\| gfx1250 \/ single \| 0 \| 3 \| 0 \| 0 \|/);
  assert.equal(markdown.match(/\+50.0%/g).length, 5);
});

test('refuses a missing candidate or mismatched measured commit', () => {
  const data = adaptPublishedDashboardData(dataset(publication())).data;
  assert.throws(() => buildBenchmarkSummary(data, 'missing', candidateSha), /does not match/);
  assert.throws(() => buildBenchmarkSummary(data, 'candidate', baselineSha), /does not match/);
});

test('CLI reads the publication snapshot and writes an identified summary artifact', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'rocjitsu-summary-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const source = dataset(publication());
  await mkdir(join(directory, 'runs/default-branch'), { recursive: true });
  await writeFile(join(directory, 'index.json'), JSON.stringify(source.index));
  for (const entry of source.runs) await writeFile(join(directory, publishedRunPath(entry)), JSON.stringify(entry));
  const script = fileURLToPath(new URL('../../../emulation/rocjitsu/website/dashboard/scripts/benchmark-summary.mjs', import.meta.url));
  const output = join(directory, 'summary.json');
  execFileSync(process.execPath, [script, '--data-dir', directory, '--run-id', 'candidate', '--head-sha', candidateSha, '--output', output]);
  const summary = JSON.parse(await readFile(output, 'utf8'));
  assert.equal(summary.runId, 'candidate');
  assert.equal(summary.headSha, candidateSha);
  assert.equal(summary.baselineRunId, 'baseline');
  source.index.runFiles.push('../escape.json');
  await writeFile(join(directory, 'index.json'), JSON.stringify(source.index));
  await assert.rejects(readBenchmarkSummary(directory, 'candidate', candidateSha), /Invalid run filename/);
});


test('a zero baseline is unavailable rather than a measured speedup', () => {
  const runs = publication();
  runs[0].benchmark_results[0].timing_results_s = [0];
  const { markdown } = summarize(runs);
  assert.match(markdown, /\| gfx950 \/ default \| 1 \| 0 \| 1 \| 1 \|/);
  assert.doesNotMatch(markdown, /Infinity/);
});
