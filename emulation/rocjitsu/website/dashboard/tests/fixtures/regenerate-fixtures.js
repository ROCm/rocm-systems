import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { createFeedbackPublication } from './feedback-publication.js';
import { createOverviewContractPublication } from './overviewContractDataset.js';
import { writeFixturePublication } from './maintainedDataset.js';

export async function regenerateFixtureDirectory(directory, { overviewContract = false } = {}) {
  const publication = overviewContract ? createOverviewContractPublication() : createFeedbackPublication();
  adaptPublishedDashboardData(publication);
  return writeFixturePublication(directory, publication);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const [destination, scenario, ...extra] = process.argv.slice(2);
  if (destination?.startsWith('--') || (scenario && scenario !== '--overview-contract') || extra.length) {
    throw new Error('Usage: node tests/fixtures/regenerate-fixtures.js [directory] [--overview-contract]');
  }
  const directory = destination ?? fileURLToPath(new URL('./data/', import.meta.url));
  const publication = await regenerateFixtureDirectory(directory, { overviewContract: scenario === '--overview-contract' });
  console.log(`Wrote complete fictional maintained-schema test fixtures (${publication.runs.length} runs).`);
}
