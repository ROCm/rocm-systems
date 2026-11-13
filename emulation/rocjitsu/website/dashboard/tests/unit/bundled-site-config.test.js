import { expect, test } from 'vitest';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { loadDashboardDataFiles } from '../../src/data/dashboardData.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';

test('loads an index-only empty publication without fetching metadata', async () => {
  const index = { generatedAt: '2026-10-05T12:00:00Z', runFiles: [] };
  const requests = [];
  const result = await loadDashboardDataFiles({
    indexUrl: 'https://example.test/data/index.json',
    retryDelaysMs: [],
    fetch: async (url) => {
      requests.push(url);
      if (url !== 'https://example.test/data/index.json') throw new Error(`Unexpected request: ${url}`);
      return { ok: true, text: async () => JSON.stringify(index) };
    },
  });
  expect(requests).toEqual(['https://example.test/data/index.json']);
  expect(result.data.repository).toBe('https://github.com/ROCm/rocm-systems');
  expect(result.sourceData).toEqual({ index, runs: [] });
});

function publication() {
  return createMaintainedPublication();
}

test('publication normalization uses bundled site settings without manufacturing raw metadata', () => {
  const source = publication();
  const { data, sourceData } = adaptPublishedDashboardData(source);
  expect(data).toMatchObject({ schemaVersion: 1, repository: 'https://github.com/ROCm/rocm-systems', canonicalBranch: 'develop', isBeta: true });
  expect(sourceData).toEqual(source);
  expect(sourceData).not.toHaveProperty('metadata');
  expect(sourceData).not.toHaveProperty('siteConfig');
});

test.each([undefined, null, '2', 3, 99])('rejects run schema version %s even when legacy metadata declares schema 2', (version) => {
  const source = publication();
  if (version === undefined) delete source.runs[0].schema_version;
  else source.runs[0].schema_version = version;
  source.metadata = { schemaVersion: 1, repository: 'https://example.test/legacy', isBeta: false, canonicalBranch: 'develop' };
  expect(() => adaptPublishedDashboardData(source)).toThrow(version === undefined ? /unsupported schema version/i : /unsupported schema version/i);
});

test.each([undefined, null, '2', 1, 99])('ignores obsolete index schema version %s', (version) => {
  const source = publication();
  if (version !== undefined) source.index.schemaVersion = version;
  const { data, sourceData } = adaptPublishedDashboardData(source);
  expect(data.allRuns).toHaveLength(source.runs.length);
  expect(sourceData.index).toEqual(source.index);
});

test('site settings cannot be overridden by publication extensions and are not included in raw exports', () => {
  const source = publication();
  Object.assign(source.index, { repository: 'https://example.test/untrusted', isBeta: false, canonicalBranch: 'other' });
  const { data, sourceData } = adaptPublishedDashboardData(source);
  expect(data).toMatchObject({ repository: 'https://github.com/ROCm/rocm-systems', isBeta: true, canonicalBranch: 'develop' });
  expect(sourceData).toEqual(source);
});

test('explicit tool site configuration remains separate from unchanged publication provenance', () => {
  const source = publication();
  source.runs[0].provenance.rocjitsu.repository = 'https://example.test/recorded-source';
  const siteConfig = { repository: 'https://example.test/site', isBeta: false, canonicalBranch: 'develop' };
  const { data, sourceData } = adaptPublishedDashboardData({ ...source, siteConfig });
  expect(data).toMatchObject(siteConfig);
  expect(sourceData).toEqual(source);
  expect(sourceData.runs[0].provenance.rocjitsu.repository).toBe('https://example.test/recorded-source');
});
