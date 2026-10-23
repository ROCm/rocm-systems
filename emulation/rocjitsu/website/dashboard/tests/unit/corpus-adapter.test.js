import { expect, test } from 'vitest';
import raw from '../fixtures/fictional-producer-run.json' with { type: 'json' };
import { loadDashboardDataFiles } from '../../src/data/dashboardData.js';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';

function publication() {
  const run = structuredClone(raw);
  // Accepted CPU harness output: fictional measurements, not native benchmarks.
  Object.assign(run.publication, { id: 'fictional-producer', source: { branch: 'develop', message: null, base: null, pull_request: { number: 12 } } });
  return { index: { generatedAt: '2026-10-21T20:00:00Z', runFiles: ['runs/default-branch/fictional-producer.json'] }, runs: [run] };
}

test.each([null, undefined, 42, {}, '', '   '])('rejects unusable required result name %j before exposing a loaded dataset', async (name) => {
  const source = publication();
  source.runs[0].benchmark_results[0].name = name;
  expect(() => adaptPublishedDashboardData(source)).toThrow(/result name/i);
  await expect(loadDashboardDataFiles({
    indexUrl: 'https://fictional.test/data/index.json', retryDelaysMs: [],
    fetch: async (url) => ({ ok: true, text: async () => JSON.stringify(
      new URL(url).pathname.endsWith('/index.json') ? source.index : source.runs[0],
    ) }),
  })).rejects.toThrow(/result name/i);
});

test('retains a usable published result name verbatim', () => {
  const source = publication();
  source.runs[0].benchmark_results[0].name = '  Exact published name  ';
  expect(adaptPublishedDashboardData(source).data.runs[0].tests[0].name).toBe('  Exact published name  ');
});

test('flat measurements preserve ordered timings and diagnostics with completed-only medians', () => {
  const source = publication();
  const rows = source.runs[0].benchmark_results;
  rows[0].timing_results_s = [3, 1, 2, 4];
  Object.assign(rows[1], { status: 'failed', timing_results_s: [9], exit_code: 7, error: 'fictional failure' });
  Object.assign(rows[2], { status: 'timeout', timing_results_s: [8], exit_code: null, error: 'fictional timeout' });
  const before = structuredClone(source);
  const tests = adaptPublishedDashboardData(source).data.runs[0].tests;
  expect(tests[0]).toMatchObject({ durationSeconds: 2.5, timing_results_s: [3, 1, 2, 4], testId: `gfx950/default:${rows[0].id}`, logicalTestId: rows[0].id, mode: 'default', problem: rows[0].problem });
  expect(tests.slice(1).map(({ durationSeconds }) => durationSeconds)).toEqual([null, null]);
  expect(tests[1]).toMatchObject({ exit_code: 7, error: 'fictional failure' });
  expect(source).toEqual(before);
  rows[0].timing_results_s = [];
  expect(adaptPublishedDashboardData(source).data.runs[0].tests[0]).toMatchObject({ durationSeconds: null, unavailableReason: expect.any(String) });
});

test('dataset derives one first-occurrence lookup and preserves complete source envelopes', () => {
  const source = publication();
  const first = source.runs[0];
  const second = structuredClone(first);
  second.publication.id = 'fictional-second';
  second.benchmark_results = [second.benchmark_results[0]];
  // Later definitions must not be read, rebuilt or compared for the same suite.
  Object.defineProperty(second.benchmark_results[0], 'problem', { get() { throw new Error('lookup rebuilt'); }, enumerable: false });
  source.runs.push(second);
  source.index.runFiles.push('runs/default-branch/fictional-second.json');
  const { data, sourceData } = adaptPublishedDashboardData(source);
  expect(data.catalogs.nightly.tests).toHaveLength(3);
  expect(data.catalogs.nightly.configurations['gfx950/default']).toEqual(first.benchmark_results.map(({ id }) => id));
  expect(data.testCatalog).toHaveLength(3);
  expect(data.targets).toEqual(['gfx950']);
  expect(data.runs[1].tests).toHaveLength(1);
  expect(sourceData).toEqual({ index: source.index, runs: source.runs });
  expect(sourceData.runs[0]).toBe(first);
});

test('corpus publication maps optional facts independently without modifying raw provenance', () => {
  const source = publication();
  source.runs[0].provenance.rocjitsu.rocjitsu_commit_timestamp = null;
  const before = structuredClone(source);
  const { data } = adaptPublishedDashboardData(source);
  expect(data.runs[0]).toMatchObject({ runId: 'fictional-producer', branch: 'develop', machineId: null, trigger: null, commitTimestamp: null, pullRequest: { number: 12 }, provenance: { rocjitsuCommitSha: raw.provenance.rocjitsu.rocjitsu_commit_sha } });
  expect(data.runs[0].sourceBase).toBeUndefined();
  expect(data.runs[0].provenance.details).toContainEqual({ key: 'rocjitsu.workflow_commit_sha', label: 'Workflow commit', value: null });
  expect(source).toEqual(before);
});

test('accepted fictional producer output loads through the real resource boundary without catalog requests', async () => {
  const source = publication();
  source.runs[0].extension = { optional: null };
  source.runs[0].benchmark_results[0].diagnostic_extension = { ordered: [false, 0, null] };
  source.runs[0].publication.source.base = { branch: 'develop', commit: null };
  const requests = [];
  const { data, sourceData } = await loadDashboardDataFiles({
    indexUrl: 'https://fictional.test/data/index.json', retryDelaysMs: [],
    fetch: async (url) => {
      requests.push(url);
      const file = new URL(url).pathname.slice('/data/'.length);
      const body = file === 'index.json' ? source.index : file === source.index.runFiles[0] ? source.runs[0] : null;
      if (!body) throw new Error(`Unexpected resource ${file}`);
      return { ok: true, text: async () => JSON.stringify(body) };
    },
  });
  expect(requests).toHaveLength(2);
  expect(data.runs[0].tests).toHaveLength(3);
  expect(sourceData).toEqual(source);
});
