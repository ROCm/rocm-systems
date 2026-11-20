// Run the real synchronous projection outside the test runner's event loop.
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { selectOverviewProjection } from '../../src/data/overviewProjection.js';
import { createSyntheticDataset } from '../fixtures/syntheticDataset.js';

const [generatedAt, range, scenario = 'empty'] = process.argv.slice(2);
const publication = { index: { generatedAt, runFiles: [] }, runs: [] };
if (scenario !== 'empty') {
  const fixture = createSyntheticDataset(3);
  const timestamps = scenario === 'upper'
    ? ['9999-12-31T12:00:00Z', '9999-12-31T23:30:00-01:00', '9999-12-31T22:45:00-02:00']
    : ['0000-01-01T00:30:00+01:00', '0000-01-01T00:00:00Z', '0000-01-01T01:15:00+01:00'];
  publication.index.runFiles = fixture.runFiles;
  publication.runs = fixture.runFiles.map((file, index) => {
    const run = structuredClone(fixture.bodies.get(`https://dashboard.test/data/${file}`));
    run.provenance.rocjitsu.rocjitsu_commit_timestamp = timestamps[index];
    run.provenance.corpus.corpus_commit_timestamp = timestamps[index];
    run.execution_summary.timestamp = timestamps[index];
    run.execution_summary.finished_at = timestamps[index];
    return run;
  });
}
const originalPublication = JSON.stringify(publication);
const { data } = adaptPublishedDashboardData(publication);
const originalData = JSON.stringify(data);
process.send({ stage: 'validated-before-projection', generatedAt: data.generatedAt, runs: data.runs.length, range }, () => {
  const filters = scenario === 'empty' ? { targets: [], modes: [], suites: [] }
    : { targets: ['gfx1250'], modes: ['single'], suites: ['Triton'] };
  const model = selectOverviewProjection(data, filters, { range });
  const proof = scenario === 'empty' ? {} : { populated: {
    attempts: model.attempts.map((attempt) => ({ id: attempt.runId, eligible: attempt.eligible, dayKey: attempt.points[0]?.dayKey })),
    slots: model.slots.filter((slot) => slot.run).map((slot) => ({ id: slot.run.runId, dayKey: slot.dayKey })),
    cohort: model.rows[0]?.cohortIds,
    points: model.rows[0]?.points.filter((point) => point.run).map((point) => ({ id: point.runId, dayKey: point.dayKey, durationSeconds: point.durationSeconds })),
    timestamps: data.runs.map((run) => run.commitTimestamp),
    originalInputsUnchanged: originalPublication === JSON.stringify(publication) && originalData === JSON.stringify(data),
    sourceLeavesRetained: model.attempts.every((attempt) => data.runs.includes(attempt.run)
      && attempt.points.every((point) => point.run === attempt.run && point.results.every(({ result }) => attempt.run.tests.includes(result))))
      && model.rows.every((row) => row.points.filter((point) => point.run).every((point) => point.results.every(({ result }) => point.run.tests.includes(result)))),
    independentWrappers: model.rows.every((row) => row.points.filter((point) => point.run).every((point) => {
      const attempt = model.attempts.find((attempt) => attempt.run === point.run).points[0];
      return point !== attempt && point.results !== attempt.results && point.reasons !== attempt.reasons;
    })),
  } };
  process.send({ stage: 'result', anchorDay: model.anchorDay, startDay: model.startDay, dayKeys: model.dayKeys, ...proof }, () => process.disconnect());
});
