import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createFeedbackPublication } from './feedback-publication.js';
import { writeFixturePublication } from './schema2Dataset.js';

export async function regenerateFixtureDirectory(directory) {
  const publication = createFeedbackPublication();
  validatePublishedDashboardData(publication);
  return writeFixturePublication(directory, publication);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const directory = process.argv[2] ?? fileURLToPath(new URL('./data/', import.meta.url));
  await regenerateFixtureDirectory(directory);
  console.log('Wrote complete fictional schema-2 test fixtures (24 develop, 21 branches).');
}
