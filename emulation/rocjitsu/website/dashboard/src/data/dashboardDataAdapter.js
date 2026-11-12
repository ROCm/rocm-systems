import { configurationKey } from './runSchema.js';
import { DASHBOARD_SITE_CONFIG } from '../config/siteConfig.js';
import { backfillRunIds, sortRunsByCommit } from './runOrdering.js';
import { normalizePublishedData, safePublishedUrl } from '../../../../scripts/benchmark-data/publishedRuns.mjs';
export { RUN_FILE_PATTERN, safePublishedUrl, validatePublishedManifest } from '../../../../scripts/benchmark-data/publishedRuns.mjs';

export function validateDashboardSiteConfig(siteConfig) {
  if (!safePublishedUrl(siteConfig?.repository)) throw new Error('Expected dashboard site configuration to contain a safe HTTP repository URL');
  if (typeof siteConfig.isBeta !== 'boolean' || siteConfig.canonicalBranch !== 'develop') throw new Error('Expected site configuration isBeta boolean and canonicalBranch develop');
}

export function adaptPublishedDashboardData({ index, runs, runErrors = [], siteConfig = DASHBOARD_SITE_CONFIG }) {
  validateDashboardSiteConfig(siteConfig);
  const data = normalizePublishedData({ index, runs, runErrors, canonicalBranch: siteConfig.canonicalBranch });
  const catalogs = new Map();
  const definitions = new Map();
  runs.forEach((run) => {
    const suite = run.execution_summary.benchmark_suite;
    if (!catalogs.has(suite)) catalogs.set(suite, { id: suite, tests: [], configurations: {} });
    const catalog = catalogs.get(suite);
    const configurations = new Map();
    for (const row of run.benchmark_results) {
      const key = configurationKey(row.target, row.thread_policy);
      if (Object.hasOwn(catalog.configurations, key)) continue;
      if (!configurations.has(key)) configurations.set(key, []);
      configurations.get(key).push(row);
    }
    const knownTests = new Set(catalog.tests.map(({ id }) => id));
    for (const [key, rows] of configurations) {
      Object.defineProperty(catalog.configurations, key, { value: rows.map(({ id }) => id), enumerable: true });
      for (const row of rows) {
        if (knownTests.has(row.id)) continue;
        const definition = { id: row.id, suite: row.suite, name: row.name, operation: row.operation, problem: row.problem };
        knownTests.add(row.id);
        catalog.tests.push(definition);
        definitions.set(definition.id, definition);
      }
    }
  });
  return { sourceData: { index, runs }, data: {
    ...siteConfig, ...data,
    testCatalog: [...definitions.values()], catalogs: Object.fromEntries(catalogs),
    latestRun: data.runs.at(-1) ?? null,
    latestCommitRun: sortRunsByCommit(data.runs).at(-1) ?? null,
    backfillRunIds: backfillRunIds(data.runs),
  } };
}
