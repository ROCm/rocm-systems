import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createDashboardFixture } from './dashboardFixture.js';

export const {
  metadata: dataMetadata,
  index: dataIndex,
  runs: publishedRuns,
  catalogs: publishedCatalogs,
} = createDashboardFixture();
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
