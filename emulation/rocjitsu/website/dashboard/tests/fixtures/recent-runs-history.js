// Fictional, test-only attempts derived from the schema-2 publication.
import { createSchema2Publication } from './schema2Dataset.js';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';

export function createRecentRunsHistory(count = 65) {
  const publication = createSchema2Publication();
  const canonical = publication.runs.filter((run) => run.source.branch === 'develop');
  const attempts = Array.from({ length: count }, (_, index) => {
    const run = structuredClone(canonical[index % canonical.length]);
    run.id = `fictional-pagination-${String(index).padStart(3, '0')}`;
    // Rerun historical commits; completion order must win over commit order.
    run.execution.completedAt = new Date(Date.UTC(2026, 9, 5, 10, Math.floor(index / 2))).toISOString();
    run.execution.trigger = 'manual';
    return run;
  });
  publication.runs = [...attempts, ...publication.runs.filter((run) => run.source.branch !== 'develop')];
  publication.index.runFiles = publication.runs.map(({ id }) => `runs/${id}.json`);
  const data = validatePublishedDashboardData(publication).data;
  return { publication, data };
}
