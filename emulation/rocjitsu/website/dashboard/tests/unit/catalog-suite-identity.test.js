import { expect, test } from 'vitest';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { parsePublishedJson } from '../../src/data/publishedJson.js';

test.each(['missing', 'nonstring'])('suite identity rejects %s metadata', (change) => {
  const publication = createMaintainedPublication();
  for (const run of publication.runs) {
    const id = run.execution_summary.benchmark_suite;
    if (change === 'missing') delete run.execution_summary.benchmark_suite;
    else run.execution_summary.benchmark_suite = change === 'nonstring' ? [id] : 'Other';
  }
  expect(() => adaptPublishedDashboardData(publication)).toThrow(/benchmark suite/);
});

test('strict JSON boundary rejects duplicate suite identity before it can shadow the declaration', () => {
  expect(() => parsePublishedJson('{"benchmark_suite":"first","benchmark_suite":"second"}')).toThrow(/Duplicate JSON key/);
});
