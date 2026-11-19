import raw from '../fixtures/fictional-producer-run.json' with { type: 'json' };

export function readMaintainedExamplePublication() {
  const run = structuredClone(raw);
  run.publication.id = 'run-001';
  run.publication.source.branch = 'develop';
  return { index: { generatedAt: '2026-10-21T20:00:00Z', runFiles: ['runs/default-branch/run-001.json'] }, runs: [run] };
}
