import { readFileSync } from 'node:fs';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
export { createFeedbackPublication } from './feedback-publication.js';

const dataDirectory = new URL('./data/', import.meta.url);
const readJson = (name) => JSON.parse(readFileSync(new URL(name, dataDirectory), 'utf8'));

export const dataIndex = readJson('index.json');
export const publishedRuns = dataIndex.runFiles.map(readJson);
export const publishedRunErrors = publishedRuns.map(() => null);

export const publishedResult = adaptPublishedDashboardData({
  index: dataIndex,
  runs: publishedRuns,
  runErrors: publishedRunErrors,
});

export const benchmarkData = publishedResult.data;

// A canonical-only clone for scoped mutation tests; do not reseed from derived run collections.
export function cloneBenchmarkData() {
  const cloned = structuredClone(benchmarkData);
  delete cloned.allRuns;
  return cloned;
}
