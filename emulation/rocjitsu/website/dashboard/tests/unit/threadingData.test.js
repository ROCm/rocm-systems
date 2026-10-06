import { expect, test } from 'vitest';
import {
  isLegacyRun,
  selectThreadingModeData,
  validatePublishedDashboardData,
} from '../../src/data/dashboardValidation.js';
import { createSyntheticDataset } from '../fixtures/syntheticDataset.js';

function fixture(count = 3) {
  const dataset = createSyntheticDataset(count);
  return {
    metadata: dataset.bodies.get(dataset.metadataUrl),
    index: dataset.bodies.get(dataset.indexUrl),
    runs: dataset.runFiles.map((file) => dataset.bodies.get(`https://dashboard.test/data/${file}`)),
    catalogs: { [dataset.catalogPath]: dataset.bodies.get(`https://dashboard.test/data/${dataset.catalogPath}`) },
  };
}

test('only plain objects without an own mode are legacy', () => {
  expect(isLegacyRun({})).toBe(true);
  for (const value of [null, [], new Date(), 'legacy', 1, { threadingMode: null }, { threadingMode: undefined }]) {
    expect(isLegacyRun(value)).toBe(false);
  }
});

test('legacy runs cannot affect catalogs, commits, plugin groups, or publication policy', () => {
  const input = fixture();
  delete input.runs[0].threadingMode;
  input.runs[0].testCatalog = 'test-catalogs/missing.json';
  input.runs[0].source.branch = 'legacy-branch';
  input.runs[0].source.commit = input.runs[1].source.commit;
  input.runs[0].plugin.id = 'asan';
  const { data, sourceData } = validatePublishedDashboardData(input);
  expect(data.runs).toHaveLength(2);
  expect(sourceData.index.runFiles).toEqual(input.index.runFiles.slice(1));
  expect(sourceData.runs).toEqual(input.runs.slice(1));
  expect(Object.keys(sourceData.catalogs)).toEqual(Object.keys(input.catalogs));
});

test('legacy-only input yields an empty supported dataset and export', () => {
  const input = fixture();
  input.runs.forEach((run) => { delete run.threadingMode; });
  input.catalogs = {};
  const { data, sourceData } = validatePublishedDashboardData(input);
  expect(data.runs).toEqual([]);
  expect(data.latestRun).toBeNull();
  expect(sourceData.index.runFiles).toEqual([]);
  expect(sourceData.catalogs).toEqual({});
});

test.each([null, '', 'threads8', 1, undefined])('rejects explicit invalid mode %s', (mode) => {
  const input = fixture(1);
  input.runs[0].threadingMode = mode;
  expect(() => validatePublishedDashboardData(input)).toThrow('invalid threadingMode');
});

test('legacy runs still enforce filenames, duplicate paths, and load errors', () => {
  const input = fixture(1);
  delete input.runs[0].threadingMode;
  input.index.runFiles.push(input.index.runFiles[0]);
  input.runs.push(input.runs[0]);
  expect(() => validatePublishedDashboardData(input)).toThrow('Duplicate run filename');
  input.index.runFiles.pop();
  input.runs.pop();
  input.runs[0].id = 'wrong-file';
  expect(() => validatePublishedDashboardData(input)).toThrow('must be published as');
  input.runErrors = [new Error('JSON parse failed')];
  expect(() => validatePublishedDashboardData(input)).toThrow('JSON parse failed');
});

test('a supported plugin cannot use a legacy Vanilla baseline', () => {
  const input = fixture(2);
  delete input.runs[0].threadingMode;
  input.runs[1].plugin.id = 'asan';
  input.runs[1].comparisonId = input.runs[0].comparisonId;
  expect(() => validatePublishedDashboardData(input)).toThrow('without a Vanilla baseline');
});

test('strict plugin groups require identical threading modes', () => {
  const input = fixture(2);
  const pluginId = input.runs[1].id;
  input.runs[1] = structuredClone(input.runs[0]);
  input.runs[1].id = pluginId;
  input.runs[1].threadingMode = 'single';
  input.runs[1].plugin.id = 'asan';
  expect(() => validatePublishedDashboardData(input)).toThrow('does not match comparison');
});

test('mode selection rebuilds latest, catalog, suites, targets, and backfills', () => {
  const input = fixture(3);
  const single = input.runs[1];
  single.threadingMode = 'single';
  single.execution.completedAt = '2027-01-01T00:00:00.000Z';
  const catalog = structuredClone(Object.values(input.catalogs)[0]);
  catalog.id = 'single';
  catalog.tests = [catalog.tests[0]];
  catalog.targets = { gfx950: [catalog.tests[0].id] };
  input.catalogs['test-catalogs/single.json'] = catalog;
  single.testCatalog = 'test-catalogs/single.json';
  single.targets = [single.targets.find((target) => target.id === 'gfx950')];
  single.targets[0].results = [single.targets[0].results[0]];
  // Default history deliberately changes allocation without splitting its timeline.
  input.runs[2].environment.push({ key: 'engines', label: 'Engines', value: 32 });
  const { data } = validatePublishedDashboardData(input);
  const defaultData = selectThreadingModeData(data, 'default');
  const singleData = selectThreadingModeData(data, 'single');
  expect(defaultData.runs.map((run) => run.runId)).toEqual([input.runs[0].id, input.runs[2].id]);
  expect(defaultData.latestRun.runId).toBe(input.runs[2].id);
  expect(defaultData.backfillRunIds.size).toBe(0);
  expect(singleData.runs.map((run) => run.runId)).toEqual([single.id]);
  expect(singleData.latestCommitRun.runId).toBe(single.id);
  expect(singleData.targets).toEqual(['gfx950']);
  expect(singleData.testCatalog).toHaveLength(1);
  expect(selectThreadingModeData(validatePublishedDashboardData(fixture(1)).data, 'single').runs).toEqual([]);
});

test('mode selection accepts the metadata-free loading/error placeholder', () => {
  const empty = { schemaVersion: null, pluginRuns: [], runs: [], testCatalog: [] };
  expect(selectThreadingModeData(empty, 'default')).toBe(empty);
});
