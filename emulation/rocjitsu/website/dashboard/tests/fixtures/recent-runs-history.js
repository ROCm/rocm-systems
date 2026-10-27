import { fixtureRunPath } from './runPath.js';
// Fictional, test-only attempts derived from the maintained schema-1 publication.
import { createMaintainedPublication } from './maintainedDataset.js';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';

export function createRecentRunsHistory(count = 65) {
  const publication = createMaintainedPublication();
  const canonical = publication.runs.filter((run) => run.publication.source.branch === 'develop');
  const attempts = Array.from({ length: count }, (_, index) => {
    const run = structuredClone(canonical[index % canonical.length]);
    run.publication.id = `fictional-pagination-${String(index).padStart(3, '0')}`;
    // Rerun historical commits; completion order must win over commit order.
    run.execution_summary.finished_at = new Date(Date.UTC(2026, 9, 5, 10, Math.floor(index / 2))).toISOString();
    run.publication.trigger = 'manual';
    return run;
  });
  publication.runs = [...attempts, ...publication.runs.filter((run) => run.publication.source.branch !== 'develop')];
  publication.index.runFiles = publication.runs.map(fixtureRunPath);
  const data = adaptPublishedDashboardData(publication).data;
  return { publication, data };
}
