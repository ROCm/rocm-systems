import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { selectOverviewProjection } from '../../src/data/overviewProjection.js';
import BenchmarkDifferences from '../../src/components/branch/BenchmarkDifferences.jsx';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';

// Consumer contracts stay literal here so helper reuse cannot change wire data or IDs.
test('Overview mode and normalized result identities retain raw target/policy data', () => {
  const source = createMaintainedPublication();
  const before = structuredClone(source);
  const { data, sourceData } = adaptPublishedDashboardData(source);
  for (const rawRun of source.runs) {
    const run = data.allRuns.find((run) => run.runId === rawRun.publication.id);
    rawRun.benchmark_results.forEach((row, index) => {
      expect(run.tests[index]).toMatchObject({ ...row,
        testId: `${row.target}/${row.thread_policy}:${row.id}`,
        logicalTestId: row.id, mode: row.thread_policy,
      });
      expect(run.configurations).toContainEqual({ target: row.target, thread_policy: row.thread_policy });
    });
  }
  const projection = selectOverviewProjection(data, { targets: data.targets, modes: data.modes, suites: data.suites }, { range: 'ALL' });
  expect(projection.rows.map(({ key }) => key).sort()).toEqual(['gfx1250/default', 'gfx1250/single', 'gfx950/default', 'gfx950/single']);
  for (const row of projection.rows) {
    expect(row.key).toBe(`${row.target}/${row.mode}`);
    expect(['single', 'default']).toContain(row.mode);
    expect(row.cohortIds.length).toBeGreaterThan(0);
  }
  expect(sourceData).toEqual(before);
  expect(source).toEqual(before);
});

test.each([
  { branch: 'develop', directory: 'default-branch', wrongDirectory: 'side-branches' },
  { branch: 'users/fictional/topic', directory: 'side-branches', wrongDirectory: 'default-branch' },
])('indexed path keeps exact ID and directory checks for $branch', ({ branch, directory, wrongDirectory }) => {
  const source = createMaintainedPublication();
  const run = structuredClone(source.runs[0]);
  run.publication.source.branch = branch;
  const publication = { ...source, runs: [run], index: { ...source.index, runFiles: [`runs/${directory}/${run.publication.id}.json`] } };
  expect(adaptPublishedDashboardData(publication).data.allRuns[0].branch).toBe(branch);
  for (const path of [`runs/${wrongDirectory}/${run.publication.id}.json`, `runs/${directory}/fictional-wrong-id.json`]) {
    expect(() => adaptPublishedDashboardData({ ...publication, index: { ...publication.index, runFiles: [path] } }))
      .toThrow(/filename must match its ID and branch directory/);
  }
});

const configuration = { target: 'gfx1250', thread_policy: 'single' };
const result = { testId: 'gfx1250/single:work', logicalTestId: 'work', name: 'Fictional workload', suite: 'Triton', status: 'completed', durationSeconds: 10 };
test.each([
  { label: 'missing run', run: null, testResult: result, expected: 'No run selected' },
  { label: 'wrong policy', run: { configurations: [{ target: 'gfx1250', thread_policy: 'default' }] }, testResult: result, expected: 'Configuration not published' },
  { label: 'wrong target', run: { configurations: [{ target: 'gfx950', thread_policy: 'single' }] }, testResult: result, expected: 'Configuration not published' },
  { label: 'missing test', run: { configurations: [configuration] }, testResult: null, expected: 'Not in this catalog' },
  { label: 'completed', run: { configurations: [configuration] }, testResult: result, expected: 'Completed · 10.0s' },
  { label: 'timeout', run: { configurations: [configuration] }, testResult: { ...result, status: 'timeout' }, expected: 'Timeout' },
  { label: 'failed', run: { configurations: [configuration] }, testResult: { ...result, status: 'failed' }, expected: 'Failed' },
])('Branch excluded status retains $label handling', ({ run, testResult, expected }) => {
  const html = renderToStaticMarkup(createElement(BenchmarkDifferences, {
    comparison: { comparable: [], notComparable: [{ baselineTest: { ...result, status: 'failed' }, candidateTest: testResult }] },
    summary: { available: false, matched: 0, excluded: 1, reason: 'Fictional excluded result' },
    candidate: run, reference: { configurations: [configuration] }, selection: { target: 'gfx1250', mode: 'single' },
    suites: ['Triton'], suiteOptions: ['Triton'], query: '',
  }));
  expect(html).toContain(`Reference: Failed · Candidate: ${expected}`);
});
