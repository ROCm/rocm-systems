import { fixtureRunPath } from './runPath.js';
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

