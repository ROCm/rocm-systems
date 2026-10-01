import { readFileSync } from 'node:fs';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';

const dataDirectory = new URL('./data/', import.meta.url);
const readJson = (name) => JSON.parse(readFileSync(new URL(name, dataDirectory), 'utf8'));

export const dataMetadata = readJson('metadata.json');
export const dataIndex = readJson('index.json');
export const publishedRuns = dataIndex.runFiles.map(readJson);
export const publishedCatalogs = Object.fromEntries([...new Set(
  publishedRuns.map((run) => run.testCatalog),
)].map((name) => [name, readJson(name)]));
export const publishedRunErrors = publishedRuns.map(() => null);

export const publishedResult = validatePublishedDashboardData({
  metadata: dataMetadata,
  index: dataIndex,
  runs: publishedRuns,
  runErrors: publishedRunErrors,
  catalogs: publishedCatalogs,
});

export const benchmarkData = publishedResult.data;

// `loadDashboardData` rebuilds `pluginRuns` from `runs`, so a clone destined for it must drop the
// derived field to avoid re-seeding the loader with already-normalized plugin runs.
export function cloneBenchmarkData() {
  const cloned = structuredClone(benchmarkData);
  delete cloned.pluginRuns;
  return cloned;
}
