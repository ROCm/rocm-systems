// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile, writeFile, mkdir, mkdtemp, rm, cp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { normalizePublishedData } from '../../../emulation/rocjitsu/scripts/benchmark-data/index.mjs';
import { publishedRunPath } from '../../../emulation/rocjitsu/scripts/benchmark-data/runSchema.mjs';
import { buildBenchmarkSummary, readBenchmarkSummary } from '../rocjitsu_benchmark_report.js';

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

async function summarize(runs = publication()) {
  return buildBenchmarkSummary(normalizePublishedData(dataset(runs)), 'candidate', candidateSha);
}

test('uses dashboard comparisons for counts, timing rows and exact run links', async () => {
  const summary = await summarize();
  assert.equal(summary.baselineRunId, 'baseline');
  assert.match(summary.markdown, /3\/3 benchmarks completed/);
  assert.match(summary.markdown, /gfx950 \/ default: 1 faster, 1 slower, 1 within band/);
  assert.match(summary.markdown, /\| 100 s \| 120 s \| \+20.0% \|/);
  assert.match(summary.markdown, /\| 100 s \| 80 s \| -20.0% \|/);
  assert.match(summary.markdown, /\*\*1 faster, 1 slower\*\*/);
  assert.doesNotMatch(summary.markdown, /Lower duration|statistical significance/);
  const url = new URL(summary.markdown.match(/Compared with \[.*?\]\(([^)]+)\)/)[1]);
  assert.equal(url.searchParams.get('compareBaseline'), 'baseline');
  assert.equal(url.searchParams.get('compareCandidate'), 'candidate');
  assert.equal(url.searchParams.get('view'), 'compare');
});

test('prefers the recorded parent over a more recent unrelated develop result', async () => {
  const runs = publication();
  runs.push(run('other', 'd'.repeat(40), '2026-11-01T12:00:00Z'));
  assert.equal((await summarize(runs)).baselineRunId, 'baseline');
});

test('does not compare against an earlier develop run when the recorded base is missing', async () => {
  const runs = publication();
  runs[1].publication.source.base.commit = 'd'.repeat(40);
  runs.push(run('previous-attempt', candidateSha, '2026-11-01T12:00:00Z'));
  const summary = await summarize(runs);
  assert.equal(summary.baselineRunId, null);
  assert.match(summary.markdown, /Timing comparison unavailable/);
  assert.doesNotMatch(summary.markdown, /faster|slower|Duration change|Compared with|earlier develop/);
});

test('does not invent a comparison when no baseline exists', async () => {
  const summary = await summarize(publication().slice(1));
  assert.equal(summary.baselineRunId, null);
  assert.match(summary.markdown, /3\/3 benchmarks completed/);
  assert.match(summary.markdown, /Timing comparison unavailable/);
  assert.doesNotMatch(summary.markdown, /Duration change \|/);
});

test('separates failed and timed-out results from comparable timing changes', async () => {
  const runs = publication();
  runs[1].benchmark_results[0].status = 'failed';
  runs[1].benchmark_results[1].status = 'timeout';
  const { markdown } = await summarize(runs);
  assert.match(markdown, /1\/3 benchmarks completed.*1 failed.*1 timed out/);
  assert.match(markdown, /gfx950 \/ default: 0 faster, 0 slower, 1 within band, 2 not comparable/);
  assert.doesNotMatch(markdown, /\+20.0%/);
});

test('flags changed environment, machine, dependencies and workload inputs', async () => {
  const runs = publication();
  runs[1].provenance.machine.cpu = 'other-cpu';
  runs[1].publication.machine_label = 'other-host';
  runs[1].publication.tensile_commit_sha = 'd'.repeat(40);
  runs[1].benchmark_results[0].problem.rows += 1;
  const { markdown } = await summarize(runs);
  assert.match(markdown, /Changed inputs:.*environment.*machine.*Tensile.*workload/);
  assert.match(markdown, /Timing differences may reflect these changes/);
});

test('escapes untrusted benchmark names in Markdown tables', async () => {
  const runs = publication();
  runs[1].benchmark_results[0].name = '<img> | [link](url)\n@someone';
  const { markdown } = await summarize(runs);
  assert.ok(markdown.includes('&lt;img&gt; \\| \\[link\\]\\(url\\) &#64;someone'));
  assert.ok(!markdown.includes('<img>'));
});

test('uses medians from the dashboard adapter instead of the first sample', async () => {
  const runs = publication();
  runs[1].benchmark_results[0].timing_results_s = [1, 120, 130];
  assert.match((await summarize(runs)).markdown, /\| 100 s \| 120 s \| \+20.0% \|/);
});

test('groups targets and thread policies separately and limits the largest changes', async () => {
  const runs = publication();
  for (const entry of runs) {
    const extra = entry.benchmark_results.map((row) => ({ ...structuredClone(row), target: 'gfx1250', thread_policy: 'single' }));
    entry.benchmark_results.push(...extra);
    if (entry.publication.id === 'candidate') {
      for (const row of entry.benchmark_results) row.timing_results_s = [150];
    }
  }
  const { markdown } = await summarize(runs);
  assert.match(markdown, /gfx950 \/ default: 0 faster, 3 slower, 0 within band/);
  assert.match(markdown, /gfx1250 \/ single: 0 faster, 3 slower, 0 within band/);
  assert.equal(markdown.match(/\+50.0%/g).length, 5);
});

test('refuses a missing candidate or mismatched measured commit', async () => {
  const data = normalizePublishedData(dataset(publication()));
  await assert.rejects(buildBenchmarkSummary(data, 'missing', candidateSha), /does not match/);
  await assert.rejects(buildBenchmarkSummary(data, 'candidate', baselineSha), /does not match/);
});

test('CLI produces the summary without a website checkout', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'rocjitsu-summary-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const source = dataset(publication());
  await mkdir(join(directory, 'runs/default-branch'), { recursive: true });
  await writeFile(join(directory, 'index.json'), JSON.stringify(source.index));
  for (const entry of source.runs) await writeFile(join(directory, publishedRunPath(entry)), JSON.stringify(entry));
  const checkout = join(directory, 'checkout');
  const scripts = join(checkout, '.github/scripts');
  await mkdir(scripts, { recursive: true });
  for (const file of ['rocjitsu_benchmark_report.js', 'rocjitsu_fuzz_report.js']) {
    await cp(new URL(`../${file}`, import.meta.url), join(scripts, file));
  }
  const sharedData = join(checkout, 'emulation/rocjitsu/scripts/benchmark-data');
  await mkdir(sharedData, { recursive: true });
  await cp(new URL('../../../emulation/rocjitsu/scripts/benchmark-data/', import.meta.url), sharedData, { recursive: true });
  const script = join(scripts, 'rocjitsu_benchmark_report.js');
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


test('a zero baseline is unavailable rather than a measured speedup', async () => {
  const runs = publication();
  runs[0].benchmark_results[0].timing_results_s = [0];
  const { markdown } = await summarize(runs);
  assert.match(markdown, /gfx950 \/ default: 1 faster, 0 slower, 1 within band, 1 not comparable/);
  assert.doesNotMatch(markdown, /Infinity/);
});


test('a run without a recorded base does not report timing comparisons', async () => {
  const runs = publication();
  runs[1].publication.source.base = null;
  const summary = await summarize(runs);
  assert.equal(summary.baselineRunId, null);
  assert.match(summary.markdown, /3\/3 benchmarks completed/);
  assert.doesNotMatch(summary.markdown, /faster|slower|Duration change|Compared with/);
});
