import { fixtureRunPath } from './runPath.js';
import { readFileSync } from 'node:fs';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createSchema2Publication } from './schema2Dataset.js';

// Additive v11 fixture overlay; never imported by production application code.
export function createFeedbackPublication() {
  const publication = createSchema2Publication();
  const run = structuredClone(publication.runs.find((run) => run.id === 'fictional-branch-01'));
  run.id = 'fictional-rattataking-test-branch';
  run.source = { branch: 'users/RattataKing/test-branch', commit: 'f1c7'.repeat(10),
    committedAt: '2026-10-05T09:00:00.000Z', message: 'FICTIONAL RattataKing test branch',
    base: { branch: 'develop', commit: publication.runs.find((run) => run.id === 'fictional-develop-20').source.commit },
  };
  run.execution.completedAt = '2026-10-05T11:20:00.000Z';
  publication.runs.push(run);
  publication.index.runFiles.push(fixtureRunPath(run));
  return publication;
}

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
