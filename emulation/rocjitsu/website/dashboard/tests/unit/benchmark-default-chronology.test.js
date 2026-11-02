import { expect, test } from 'vitest';
import { selectBenchmarkCatalog, selectBenchmarkSeries, selectDefaultBenchmarks } from '../../src/data/selectors.js';

const filters = { targets: ['gfx950'], suites: ['Suite'], modes: ['single'] };
const result = (id) => ({ logicalTestId: id, target: 'gfx950', mode: 'single', suite: 'Suite', status: 'completed', durationSeconds: 0 });

test('measured defaults skip undated-only workloads without removing them from the catalog', () => {
  const data = {
    testCatalog: ['undated-a', 'undated-b', 'undated-c', 'dated'].map((id) => ({ id, name: id, suite: 'Suite' })),
    runs: [
      { runId: 'undated', commitTimestamp: null, tests: ['undated-a', 'undated-b', 'undated-c'].map(result) },
      { runId: 'dated', commitTimestamp: '2026-10-01T00:00:00Z', provenance: { rocjitsuCommitSha: 'abc' }, tests: [result('dated')] },
    ],
  };
  const before = structuredClone(data);
  const defaults = selectDefaultBenchmarks(data, filters);
  expect(defaults.map(({ id }) => id)).toEqual(['dated']);
  expect(selectBenchmarkSeries(data, filters, defaults[0].id).series[0].points[0].value).toBe(0);
  expect(selectBenchmarkCatalog(data, filters).available).toHaveLength(4);
  expect(selectDefaultBenchmarks({ ...data, runs: [data.runs[0]] }, filters)).toEqual([]);
  expect(data).toEqual(before);
});
