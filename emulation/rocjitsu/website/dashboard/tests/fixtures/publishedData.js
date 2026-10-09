import { readFileSync } from 'node:fs';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
export { createFeedbackPublication } from './feedback-publication.js';

const dataDirectory = new URL('./data/', import.meta.url);
const readJson = (name) => JSON.parse(readFileSync(new URL(name, dataDirectory), 'utf8'));

export const dataIndex = readJson('index.json');
export const publishedRuns = dataIndex.runFiles.map(readJson);
export const publishedCatalogs = Object.fromEntries([...new Set(
  publishedRuns.map((run) => run.testCatalog),
)].map((name) => [name, readJson(name)]));
export const publishedRunErrors = publishedRuns.map(() => null);

export const publishedResult = validatePublishedDashboardData({
  index: dataIndex,
  runs: publishedRuns,
  runErrors: publishedRunErrors,
  catalogs: publishedCatalogs,
});

export const benchmarkData = publishedResult.data;

// A canonical-only clone for scoped mutation tests; do not reseed from derived run collections.
export function cloneBenchmarkData() {
  const cloned = structuredClone(benchmarkData);
  delete cloned.allRuns;
  return cloned;
}
