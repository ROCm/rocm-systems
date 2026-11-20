import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from 'vitest';
import { loadDashboardDataDirectory } from '../../scripts/process-dashboard-data.mjs';
import { selectOverviewProjection } from '../../src/data/overviewProjection.js';
import { createFeedbackPublication } from '../fixtures/feedback-publication.js';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';

test('default fictional trends include three slowdown attempts below baseline in every visible range', () => {
  const { data } = adaptPublishedDashboardData(createFeedbackPublication());
  const filters = { targets: data.targets, modes: data.modes, suites: data.suites };
  for (const range of ['1W', '1M', '3M', 'ALL']) {
    const model = selectOverviewProjection(data, filters, { range });
    const slowdowns = new Set(model.rows.flatMap((row) => row.points.filter((point) => Number.isFinite(point.speedup) && point.speedup < 1).map((point) => point.runId)));
    expect([...slowdowns].sort()).toEqual(['fictional-develop-18', 'fictional-develop-20', 'fictional-develop-22']);
  }
});

const checkedIn = fileURLToPath(new URL('../fixtures/data/', import.meta.url));
const generator = fileURLToPath(new URL('../fixtures/regenerate-fixtures.js', import.meta.url));
const jsonFiles = async (directory) => (await readdir(directory, { recursive: true })).filter((name) => name.endsWith('.json')).sort();

test('default fixture includes an ended develop run with unexecuted cases for Recent Runs', () => {
  const publication = createFeedbackPublication();
  const interrupted = publication.runs.find((run) => run.publication.id === 'fictional-unfinished-develop');
  expect(interrupted).toBeDefined();
  expect(interrupted.publication.source.branch).toBe('develop');
  expect(typeof interrupted.execution_summary.finished_at).toBe('string');
  expect(Number.isFinite(Date.parse(interrupted.execution_summary.finished_at))).toBe(true);
  expect(interrupted.benchmark_results[0].status).toBe('completed');
  expect(interrupted.benchmark_results.slice(1).every((result) => result.status === 'failed'
    && result.exit_code === null && result.error === 'run interrupted before completion'
    && result.timing_results_s.length === 0)).toBe(true);
});

test('one regeneration command reproduces the complete checked-in publication', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'dashboard-fixture-regeneration-'));
  try {
    await writeFile(path.join(directory, 'obsolete.json'), '{}\n');
    const result = spawnSync(process.execPath, [generator, directory], { encoding: 'utf8' });
    expect(result.status, result.stderr).toBe(0);
    const files = await jsonFiles(directory);
    expect(files).toEqual(await jsonFiles(checkedIn));
    for (const name of files) {
      expect(await readFile(path.join(directory, name), 'utf8'), name).toBe(await readFile(path.join(checkedIn, name), 'utf8'));
    }
    const validated = await loadDashboardDataDirectory(directory);
    const expected = createFeedbackPublication();
    expect(validated.sourceData.runs).toHaveLength(expected.runs.length);
    expect(validated.data.runs).toHaveLength(expected.runs.filter((run) => run.publication.source.branch === 'develop').length);
    expect(validated.data.allRuns).toHaveLength(expected.runs.length);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

async function withOverviewFixture(assertions) {
  const directory = await mkdtemp(path.join(tmpdir(), 'dashboard-overview-fixture-'));
  try {
    const result = spawnSync(process.execPath, [generator, directory, '--overview-contract'], { encoding: 'utf8' });
    expect(result.status, result.stderr).toBe(0);
    const validated = await loadDashboardDataDirectory(directory);
    const { index, runs } = validated.sourceData;
    expect(runs.map(({ publication }) => publication.id)).toEqual([
      'fictional-contract-old-1', 'fictional-contract-old-2',
      'fictional-contract-middle-4', 'fictional-contract-middle-5',
      'fictional-contract-current-first', 'fictional-contract-current-retry',
      'fictional-contract-current-failed',
    ]);
    expect(index.runFiles).toHaveLength(7);
    expect(Object.keys(validated.data.catalogs)).toHaveLength(3);
    expect(runs).toEqual(await Promise.all(index.runFiles.map(async (name) => JSON.parse(await readFile(path.join(directory, name), 'utf8')))));
    await assertions(validated);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

const overviewFilters = { targets: ['gfx950', 'gfx1250'], modes: ['single'], suites: ['Triton', 'Llama'] };

test('unsupported fixture scenario fails before clearing its destination', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'dashboard-invalid-fixture-'));
  try {
    const marker = path.join(directory, 'keep.json');
    await writeFile(marker, '{}\n');
    const result = spawnSync(process.execPath, [generator, directory, '--unknown-scenario'], { encoding: 'utf8' });
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain('Usage:');
    expect(await readFile(marker, 'utf8')).toBe('{}\n');
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('contract fixture preserves typed definitions and exact target/policy membership through regeneration', async () => {
  await withOverviewFixture(({ data, sourceData }) => {
    expect(data.runs).toHaveLength(7);
    expect(data.modes).toEqual(['single', 'default']);
    const definitions = Object.fromEntries(data.catalogs['fictional-contract-current'].tests.map((row) => [row.id, row]));
    expect(definitions.a.problem).toEqual({ shape: [16, 32], tuning: { enabled: true, tile: null } });
    expect(definitions.c.problem).toEqual({ shape: '[16,32]' });
    expect(sourceData.runs.every((run) => run.publication.source.message.startsWith('FICTIONAL'))).toBe(true);
  });
});

test('contract fixture discriminates three catalogs, missing days/configurations and failed endpoint filtering', async () => {
  await withOverviewFixture(({ data, sourceData }) => {
    const original = JSON.stringify(sourceData);
    const model = selectOverviewProjection(data, overviewFilters, { range: 'ALL' });
    const single950 = model.rows.find(({ key }) => key === 'gfx950/single');
    expect(single950.catalogIds).toHaveLength(3);
    expect(single950.cohortIds).toEqual(['a']);
    expect(single950.points.map(({ durationSeconds }) => durationSeconds)).toEqual([10, 9, null, 8, 7, 4]);
    expect(model.slots[2]).toMatchObject({ dayKey: '2026-10-03', run: null });
    const single1250 = model.rows.find(({ key }) => key === 'gfx1250/single');
    expect(single1250.cohortIds).toEqual(['a', 'b']);
    expect(single1250.points[3].reason).toBe('configuration-absent');
    expect(single1250.latest).toMatchObject({ reason: 'failed', completed: 1, required: 2, durationSeconds: null, speedup: null });
    const withoutFailedFamily = selectOverviewProjection(data, { ...overviewFilters, suites: ['Triton'] }, { range: 'ALL' });
    expect(withoutFailedFamily.rows.find(({ key }) => key === 'gfx1250/single').latest.durationSeconds).toBe(4);
    const default950 = selectOverviewProjection(data, { ...overviewFilters, targets: ['gfx950'], modes: ['default'] }, { range: 'ALL' }).rows[0];
    expect(default950.cohortIds).toEqual([]);
    expect(default950.latest).toMatchObject({ reason: 'no-comparable-workloads', durationSeconds: null });
    expect(selectOverviewProjection(data, { ...overviewFilters, suites: [] }).rows.every(({ reference }) => reference === null)).toBe(true);
    expect(selectOverviewProjection(data, { ...overviewFilters, targets: [] }).rows).toEqual([]);
    const scoped = selectOverviewProjection(data, overviewFilters, { range: 'ALL', catalogId: 'fictional-contract-current' });
    expect(scoped.dayKeys).toEqual(model.dayKeys);
    expect(scoped.rows.find(({ key }) => key === 'gfx950/single').cohortIds).toEqual(['a', 'd']);
    expect(scoped.rows.find(({ key }) => key === 'gfx950/single').latest.durationSeconds).toBe(304);
    const unavailable = selectOverviewProjection(data, overviewFilters, { range: 'ALL', catalogId: 'unpublished-fictional-suite' });
    expect(unavailable.rows.every(({ cohortIds, reference }) => cohortIds.length === 0 && reference === null)).toBe(true);
    expect(JSON.stringify(sourceData)).toBe(original);
  });
});

test('contract fixture retains same-commit weekly retries but rejects incomplete all-selected-lane attempts', async () => {
  await withOverviewFixture(({ data }) => {
    const model = selectOverviewProjection(data, overviewFilters, { range: '1W' });
    expect(model.slots.filter(({ run }) => run).map(({ run }) => run.runId)).toEqual([
      'fictional-contract-old-1', 'fictional-contract-old-2',
      'fictional-contract-current-first', 'fictional-contract-current-retry',
    ]);
    expect(model.attempts.find(({ runId }) => runId === 'fictional-contract-current-failed').eligible).toBe(false);
    const attempts = data.runs.filter(({ runId }) => runId.startsWith('fictional-contract-current-'));
    expect(new Set(attempts.map(({ provenance }) => provenance.rocjitsuCommitSha)).size).toBe(1);
    expect(new Set(attempts.map(({ timestamp }) => timestamp)).size).toBe(3);
    expect(model.latestRun.runId).toBe('fictional-contract-current-retry');
    const only950 = selectOverviewProjection(data, { ...overviewFilters, targets: ['gfx950'] }, { range: '1W' });
    expect(only950.latestRun.runId).toBe('fictional-contract-current-failed');
    expect(only950.rows[0].cohortIds).toEqual(model.rows.find(({ key }) => key === 'gfx950/single').cohortIds);
  });
});
