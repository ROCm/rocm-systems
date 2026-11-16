import { fixtureRunPath } from './runPath.js';
import { createMaintainedPublication } from './maintainedDataset.js';

// Feedback v1 fixture overlay; never imported by production application code.
export function createFeedbackPublication() {
  const publication = createMaintainedPublication();
  // Deterministic fictional regressions exercise both sides of the 1.00× line.
  const slowdowns = new Map([
    ['fictional-develop-18', 'gfx1250/single'],
    ['fictional-develop-20', 'gfx950/default'],
    ['fictional-develop-22', 'gfx950/single'],
  ]);
  for (const attempt of publication.runs) {
    const configuration = slowdowns.get(attempt.publication.id);
    if (!configuration) continue;
    for (const result of attempt.benchmark_results) {
      if (`${result.target}/${result.thread_policy}` === configuration && result.status === 'completed') {
        result.timing_results_s = result.timing_results_s.map((seconds) => Number((seconds * 1.4).toFixed(2)));
      }
    }
  }
  const run = structuredClone(publication.runs.find((run) => run.publication.id === 'fictional-branch-01'));
  run.publication.id = 'fictional-rattataking-test-branch';
  run.provenance.rocjitsu.rocjitsu_commit_sha = 'f1c7'.repeat(10);
  run.provenance.rocjitsu.rocjitsu_commit_timestamp = '2026-10-05T09:00:00.000Z';
  run.publication.source = { branch: 'users/RattataKing/test-branch', message: 'FICTIONAL RattataKing test branch',
    base: { branch: 'develop', commit: publication.runs.find((run) => run.publication.id === 'fictional-develop-20').provenance.rocjitsu.rocjitsu_commit_sha },
  };
  run.execution_summary.finished_at = '2026-10-05T11:20:00.000Z';
  publication.runs.push(run);
  publication.index.runFiles.push(fixtureRunPath(run));

  const unfinished = structuredClone(publication.runs.find((run) => run.publication.id === 'fictional-develop-00'));
  unfinished.publication.id = 'fictional-unfinished-develop';
  unfinished.publication.source.message = 'FICTIONAL interrupted develop run with unexecuted cases';
  for (const result of unfinished.benchmark_results.slice(1)) {
    Object.assign(result, { status: 'failed', exit_code: null, error: 'run interrupted before completion', timing_results_s: [] });
  }
  publication.runs.push(unfinished);
  publication.index.runFiles.push(fixtureRunPath(unfinished));
  return publication;
}
