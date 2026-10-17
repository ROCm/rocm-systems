import { afterEach, expect, test, vi } from 'vitest';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

function selectTarget(version) {
  vi.resetModules();
  vi.doMock('../../src/config/metadata.json', () => ({ default: {
    schemaVersion: version,
    repository: 'https://github.com/ROCm/rocm-systems',
    isBeta: true,
    canonicalBranch: 'develop',
  } }));
}

afterEach(() => {
  vi.doUnmock('../../src/config/metadata.json');
  vi.resetModules();
});

test('metadata selects the requested run schema without being checked as a metadata format', async () => {
  selectTarget(3);
  const { DASHBOARD_SITE_CONFIG } = await import('../../src/config/siteConfig.js');
  const { CURRENT_RUN_SCHEMA_VERSION } = await import('../../src/data/runSchema.js');
  expect(DASHBOARD_SITE_CONFIG.canonicalBranch).toBe('develop');
  expect(CURRENT_RUN_SCHEMA_VERSION).toBe(3);
});

test('a requested target without an implemented migration rejects runs rather than relabeling them', async () => {
  selectTarget(3);
  const { validatePublishedDashboardData } = await import('../../src/data/dashboardValidation.js');
  expect(() => validatePublishedDashboardData(createSchema2Publication())).toThrow(/no migration.*schema 2.*schema 3/i);
});

test('the schema-1 patch produces schema 2, not an unimplemented target label', async () => {
  selectTarget(3);
  const { migrateSchema1Run } = await import('../../src/data/runSchema.js');
  const result = migrateSchema1Run({
    id: 'legacy', comparisonId: 'legacy', plugin: { id: 'vanilla', name: 'Vanilla' },
    targets: [{ id: 'gfx950', results: [{ testId: 'a', status: 'completed', durationSeconds: 1, error: null }] }],
  }, { id: 'legacy', tests: [], targets: { gfx950: ['a'] } });
  expect(result.run.schemaVersion).toBe(2);
  expect(result.run.configurations[0].threadingMode).toBe('MT');
});

test('a missing schemaVersion is always interpreted as schema 1 before shape validation', async () => {
  const { runSchemaVersion } = await import('../../src/data/runSchema.js');
  expect(runSchemaVersion({ id: 'legacy' })).toBe(1);
  expect(runSchemaVersion({ id: 'legacy', configurations: [] })).toBe(1);
  expect(() => runSchemaVersion({ id: 'legacy', schemaVersion: null })).toThrow(/unsupported schema version/);
});
