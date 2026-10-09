import { expect, test } from 'vitest';
import { loadDashboardData, validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { loadDashboardDataFiles } from '../../src/data/dashboardData.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

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
  expect(result.sourceData).toEqual({ index, catalogs: {}, runs: [] });
});

function publication() {
  return createSchema2Publication();
}

test('publication normalization uses bundled site settings without manufacturing raw metadata', () => {
  const source = publication();
  const { data, sourceData } = validatePublishedDashboardData(source);
  expect(data).toMatchObject({ schemaVersion: 2, repository: 'https://github.com/ROCm/rocm-systems', canonicalBranch: 'develop', isBeta: true });
  expect(sourceData).toEqual(source);
  expect(sourceData).not.toHaveProperty('metadata');
  expect(sourceData).not.toHaveProperty('siteConfig');
});

test.each([undefined, null, '2', 3, 99])('rejects run schema version %s even when legacy metadata declares schema 2', (version) => {
  const source = publication();
  if (version === undefined) delete source.runs[0].schemaVersion;
  else source.runs[0].schemaVersion = version;
  source.metadata = { schemaVersion: 2, repository: 'https://example.test/legacy', isBeta: false, canonicalBranch: 'develop' };
  expect(() => validatePublishedDashboardData(source)).toThrow(version === undefined ? /schema-1.*contract/i : /unsupported schema version/i);
});

test.each([undefined, null, '2', 1, 99])('ignores obsolete index schema version %s', (version) => {
  const source = publication();
  if (version !== undefined) source.index.schemaVersion = version;
  const { data, sourceData } = validatePublishedDashboardData(source);
  expect(data.allRuns).toHaveLength(source.runs.length);
  expect(sourceData.index).toEqual(source.index);
});

test('site settings cannot be overridden by publication extensions and are not included in raw exports', () => {
  const source = publication();
  Object.assign(source.index, { repository: 'https://example.test/untrusted', isBeta: false, canonicalBranch: 'other' });
  const { data, sourceData } = validatePublishedDashboardData(source);
  expect(data).toMatchObject({ repository: 'https://github.com/ROCm/rocm-systems', isBeta: true, canonicalBranch: 'develop' });
  expect(sourceData).toEqual(source);
});

test('explicit tool site configuration round trips separately from unchanged publication provenance', () => {
  const source = publication();
  source.runs[0].source.repository = 'https://example.test/recorded-source';
  const siteConfig = { repository: 'https://example.test/site', isBeta: false, canonicalBranch: 'develop' };
  const { data, sourceData } = validatePublishedDashboardData({ ...source, siteConfig });
  expect(data).toMatchObject(siteConfig);
  expect(loadDashboardData(JSON.parse(JSON.stringify(data)))).toEqual(data);
  expect(sourceData).toEqual(source);
  expect(sourceData.runs[0].source.repository).toBe('https://example.test/recorded-source');
});
