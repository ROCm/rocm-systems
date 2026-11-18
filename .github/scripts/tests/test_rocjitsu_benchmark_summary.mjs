// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile, writeFile, mkdir, mkdtemp, rm, cp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { buildBenchmarkSummary, readBenchmarkSummary } from '../rocjitsu_benchmark_report.js';

const fixture = JSON.parse(await readFile(new URL('../../../emulation/rocjitsu/website/dashboard/tests/fixtures/fictional-producer-run.json', import.meta.url)));
const candidateSha = 'c'.repeat(40);
const baselineSha = 'b'.repeat(40);

function run(id, sha, timestamp) {
  const result = structuredClone(fixture);
  result.publication.id = id;
  result.publication.source = { branch: 'develop', base: null };
  result.provenance.rocjitsu.rocjitsu_commit_sha = sha;
  result.execution_summary.timestamp = timestamp;
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

function publishedRunPath(run) {
  const group = run.publication.source.branch === 'develop' ? 'default-branch' : 'side-branches';
  return `runs/${group}/${run.publication.id}.json`;
}

function dataset(runs) {
  return { index: { generatedAt: '2026-11-03T00:00:00Z', runFiles: runs.map((entry) => publishedRunPath(entry)) }, runs };
}

async function summarize(runs = publication()) {
  return buildBenchmarkSummary(runs, 'candidate', candidateSha);
}

test('reports run coverage, timing changes and exact run links', async () => {
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
  assert.match(summary.markdown, /\*\*1W reference:\*\*/);
  assert.doesNotMatch(summary.markdown, /dashboard Overview/);
  assert.deepEqual(url.searchParams.getAll('compareTargets'), ['gfx950']);
  assert.deepEqual(url.searchParams.getAll('compareSuites'), ['GPT-OSS', 'Triton']);
  assert.deepEqual(url.searchParams.getAll('compareModes'), ['default']);
});

test('comparison links include baseline-only and candidate-only filter options', async () => {
  const runs = publication();
  runs[0].benchmark_results.push({ ...structuredClone(runs[0].benchmark_results[0]),
    target: 'gfx1250', thread_policy: 'single', suite: 'baseline-only' });
  runs[1].benchmark_results.push({ ...structuredClone(runs[1].benchmark_results[0]),
    target: 'gfx1201', suite: 'candidate-only & extra' });
  const { markdown } = await summarize(runs);
  const url = new URL(markdown.match(/Compared with \[.*?\]\(([^)]+)\)/)[1]);
  assert.deepEqual(url.searchParams.getAll('compareTargets'), ['gfx1201', 'gfx1250', 'gfx950']);
  assert.deepEqual(url.searchParams.getAll('compareSuites'), ['GPT-OSS', 'Triton', 'baseline-only', 'candidate-only & extra']);
  assert.deepEqual(url.searchParams.getAll('compareModes'), ['default', 'single']);
});

test('ignores container hostname changes while retaining other machine inputs', async () => {
  const runs = publication();
  runs[1].provenance.machine.hostname = 'another-container';
  const before = structuredClone(runs);
  assert.doesNotMatch((await summarize(runs)).markdown, /Changed inputs/);
  assert.deepEqual(runs, before);
  runs[1].provenance.machine.kernel = 'another-kernel';
  assert.match((await summarize(runs)).markdown, /Changed inputs:.*environment/);
});

test('discloses sampling and warmup counts for both runs', async () => {
  const runs = publication();
  assert.match((await summarize(runs)).markdown,
    /Sampling per benchmark: current 1 sample and 0 warmups; baseline 1 sample and 0 warmups/);
  runs[1].execution_summary.samples = 5;
  runs[1].execution_summary.warmups = 2;
  assert.match((await summarize(runs)).markdown,
    /Sampling per benchmark: current 5 samples and 2 warmups; baseline 1 sample and 0 warmups/);
});

for (const status of ['failed', 'timeout']) {
  test(`all-${status} results have no timing verdict`, async () => {
    const runs = publication();
    for (const result of runs[1].benchmark_results) result.status = status;
    runs[1].provenance.machine.kernel = 'another-kernel';
    const { markdown } = await summarize(runs);
    assert.match(markdown, /0\/3 benchmarks completed/);
    assert.match(markdown, /Timing comparison unavailable: no comparable benchmark timings/);
    assert.doesNotMatch(markdown, /faster|slower|outside the duration band|Duration change/);
    assert.match(markdown, /Changed inputs:.*environment/);
    assert.doesNotMatch(markdown, /Timing differences/);
  });
}

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
  assert.doesNotMatch(summary.markdown, /faster|slower|Duration change|Compared with/);
  assert.match(summary.markdown, /Dashboard link chooses its own reference.*earlier develop run/);
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

test('uses median timing instead of the first sample', async () => {
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
  const runs = publication();
  assert.throws(() => buildBenchmarkSummary(runs, 'missing', candidateSha), /does not match/);
  assert.throws(() => buildBenchmarkSummary(runs, 'candidate', baselineSha), /does not match/);
});

test('CLI produces the summary with only the bot scripts installed', async (t) => {
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

test('uses the latest published attempt for the exact base commit', async () => {
  const runs = publication();
  const retry = structuredClone(runs[0]);
  retry.publication.id = 'baseline-retry';
  retry.execution_summary.finished_at = '2026-11-01T12:00:00Z';
  for (const result of retry.benchmark_results) result.timing_results_s = [200];
  runs.push(retry);
  const summary = await summarize(runs);
  assert.equal(summary.baselineRunId, 'baseline-retry');
  assert.match(summary.markdown, /\| 200 s \| 120 s \| -40.0% \|/);
});

test('uses the even-sample median without modifying the published samples', async () => {
  const runs = publication();
  runs[1].benchmark_results[0].timing_results_s = [140, 100, 120, 160];
  const before = structuredClone(runs);
  assert.match((await summarize(runs)).markdown, /\| 100 s \| 130 s \| \+30.0% \|/);
  assert.deepEqual(runs, before);
});

for (const baseline of [1, 100]) {
  test(`changes exactly on the duration band remain within the band (baseline ${baseline})`, async () => {
    const runs = publication();
    for (const result of runs[0].benchmark_results) result.timing_results_s = [baseline];
    for (const [index, scale] of [1.03, 0.97, 1].entries()) {
      runs[1].benchmark_results[index].timing_results_s = [baseline * scale];
    }
    const { markdown } = await summarize(runs);
    assert.match(markdown, /gfx950 \/ default: 0 faster, 0 slower, 3 within band/);
    assert.match(markdown, /No comparable timing changes outside the duration band/);
    assert.doesNotMatch(markdown, /Largest observed changes|Duration change \|/);
  });

  test(`changes just outside the duration band are counted and listed (baseline ${baseline})`, async () => {
    const runs = publication();
    for (const result of runs[0].benchmark_results) result.timing_results_s = [baseline];
    for (const [index, scale] of [1.030001, 0.969999, 1].entries()) {
      runs[1].benchmark_results[index].timing_results_s = [baseline * scale];
    }
    const { markdown } = await summarize(runs);
    assert.match(markdown, /gfx950 \/ default: 1 faster, 1 slower, 1 within band/);
    assert.match(markdown, /Largest observed changes/);
    assert.equal(markdown.match(/\| [+-]3.0% \|/g).length, 2);
  });
}

test('results only in the baseline remain not comparable', async () => {
  const runs = publication();
  runs[0].benchmark_results.push({ ...structuredClone(runs[0].benchmark_results[0]), target: 'gfx1250' });
  assert.match((await summarize(runs)).markdown, /gfx1250 \/ default: 0 faster, 0 slower, 0 within band, 1 not comparable/);
});

for (const samples of [[], [0], [NaN], [Infinity], [-1], [120, null]]) {
  test(`invalid timing samples do not produce a timing change (${JSON.stringify(samples)})`, async () => {
    const runs = publication();
    runs[1].benchmark_results[0].timing_results_s = samples;
    const { markdown } = await summarize(runs);
    assert.match(markdown, /2\/3 benchmarks completed/);
    assert.match(markdown, /1 not comparable/);
    assert.doesNotMatch(markdown, /\+20.0%/);
  });
}

test('duplicate measurements are rejected rather than silently replacing a result', () => {
  const runs = publication();
  runs[1].benchmark_results.push(structuredClone(runs[1].benchmark_results[0]));
  assert.throws(() => buildBenchmarkSummary(runs, 'candidate', candidateSha), /Duplicate benchmark result/);
});


test('overview references are frozen before candidate start while exact-base comparison stays separate', async () => {
  const runs = publication();
  const latest = run('latest', 'd'.repeat(40), '2026-11-01T23:00:00Z');
  const overlapping = run('overlap', 'e'.repeat(40), '2026-11-02T00:30:00Z');
  runs[1].execution_summary.finished_at = '2026-11-02T01:00:00Z';
  runs.push(latest, overlapping);
  const summary = await summarize(runs);
  assert.equal(summary.overview.latestDevelopRunId, 'latest');
  assert.equal(summary.baselineRunId, 'baseline');
  assert.ok(summary.markdown.indexOf('### Performance overview') < summary.markdown.indexOf('### Recorded-base comparison'));
  assert.match(summary.markdown, /Latest develop reference:.*dddddddddddd/);
  assert.match(summary.markdown, /completed before the candidate measurement started/);
  const future = run('future', 'f'.repeat(40), '2026-11-03T00:00:00Z');
  assert.equal((await summarize([...runs, future])).markdown, summary.markdown);
});

test('overview uses a single intersection and exposes matched coverage for all three durations', async () => {
  const runs = publication();
  runs[1].benchmark_results.pop();
  const summary = await summarize(runs);
  assert.match(summary.markdown, /\| gfx950 \| default \| 2\/3 \| 200 s \| 200 s \/ 1.000× \| 200 s \/ 1.000× \|/);
});

test('failed matched workloads make totals unavailable instead of making the suite faster', async () => {
  const runs = publication();
  runs[1].benchmark_results[0].status = 'failed';
  const { markdown } = await summarize(runs);
  assert.match(markdown, /\| gfx950 \| default \| 3\/3 \| N\/A \| 300 s \/ N\/A \| 300 s \/ N\/A \|/);
});

test('missing measurement chronology leaves overview references unavailable', async () => {
  const runs = publication();
  delete runs[1].execution_summary.timestamp;
  const summary = await summarize(runs);
  assert.deepEqual(summary.overview, { latestDevelopRunId: null, weeklyRunId: null });
  assert.match(summary.markdown, /Latest develop reference:\*\* N\/A/);
  assert.equal(summary.baselineRunId, 'baseline');
});

test('selects an eligible weekly reference from the candidate-time snapshot', async () => {
  const make = (id, sha, commit, completed, options = {}) => {
    const entry = run(id, sha.repeat(40), completed);
    entry.provenance.rocjitsu.rocjitsu_commit_timestamp = commit;
    if (options.catalog) entry.execution_summary.benchmark_suite = options.catalog;
    if (options.failed) entry.benchmark_results[0].status = 'failed';
    if (options.extra) entry.benchmark_results.push({ ...structuredClone(entry.benchmark_results[0]), id: 'extra' });
    if (options.missing) entry.benchmark_results.pop();
    if (options.invalid) entry.benchmark_results[0].timing_results_s = [0];
    return entry;
  };
  for (const [variant, expected] of Object.entries({
    ordinary: 'first', failed: 'later', invalid: 'later',
    missing: 'first', catalog: 'first', repeated: 'first',
  })) {
    const history = [
      make('outside', 'a', '2026-10-24T12:00:00Z', '2026-10-24T13:00:00Z'),
      make('first', 'b', '2026-10-25T01:00:00Z', '2026-10-25T02:00:00Z', {
        failed: variant === 'failed', invalid: variant === 'invalid', extra: variant === 'catalog',
      }),
      make('later', 'd', '2026-10-31T12:00:00Z', '2026-10-31T13:00:00Z', {
        missing: variant === 'missing', catalog: variant === 'catalog' ? 'changed' : undefined,
      }),
    ];
    if (variant === 'repeated') history.push(make('repeat', 'b', '2026-10-25T01:00:00Z', '2026-10-31T14:00:00Z'));
    const candidate = make('candidate', 'c', '2026-11-01T01:00:00Z', '2026-11-01T02:00:00Z');
    const summary = await summarize([...history, candidate]);
    assert.equal(summary.overview.weeklyRunId, expected, variant);
    assert.equal(summary.overview.latestDevelopRunId, variant === 'repeated' ? 'repeat' : 'later');
  }
});


test('overview speedup is reference duration divided by candidate duration over benchmark medians', async () => {
  const runs = publication();
  for (const result of runs[1].benchmark_results) result.timing_results_s = [25, 50, 100];
  const { markdown } = await summarize(runs);
  assert.match(markdown, /\| gfx950 \| default \| 3\/3 \| 150 s \| 300 s \/ 2.000× \| 300 s \/ 2.000× \|/);
});

test('latest develop can be failed while weekly reference stays eligible', async () => {
  const runs = publication();
  const latest = run('failed-latest', 'd'.repeat(40), '2026-11-01T12:00:00Z');
  latest.benchmark_results[0].status = 'failed';
  runs.push(latest);
  const summary = await summarize(runs);
  assert.equal(summary.overview.latestDevelopRunId, 'failed-latest');
  assert.equal(summary.overview.weeklyRunId, 'baseline');
  assert.match(summary.markdown, /\| gfx950 \| default \| 3\/3 \| 300 s \| N\/A \| 300 s \/ 1.000× \|/);
});
