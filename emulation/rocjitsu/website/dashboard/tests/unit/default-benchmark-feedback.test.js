import { expect, test } from 'vitest';
import * as selectors from '../../src/data/selectors.js';
import { readDashboardRoute } from '../../src/hooks/useDashboardState.js';
import { filterBenchmarkOptions } from '../../src/components/benchmarks/benchmarkExplorer.js';

const filters = { targets: ['gfx950'], suites: ['Suite'], modes: ['single'] };
const catalog = ['empty', 'failed', 'zero', 'measured', 'other-target', 'third', 'fourth'].map((id) => ({ id, name: 'Same name', suite: 'Suite' }));
const result = (id, durationSeconds, extra = {}) => ({ logicalTestId: id, target: 'gfx950', mode: 'single', suite: 'Suite', status: 'completed', durationSeconds, ...extra });
const data = { testCatalog: catalog, runs: [{ commitTimestamp: '2026-10-01T00:00:00Z', tests: [result('failed', null, { status: 'failed' }), result('zero', 0), result('measured', 2), result('other-target', 3, { target: 'gfx1250' }), result('third', 4), result('fourth', 5)] }] };

test('first-use benchmark defaults select three measured exact IDs, retaining zero and skipping empty or failed scopes', () => {
  expect(selectors.selectDefaultBenchmarks).toBeTypeOf('function');
  const before = structuredClone(data);
  expect(selectors.selectDefaultBenchmarks(data, filters).map((item) => item.id)).toEqual(['zero', 'measured', 'third']);
  expect(selectors.selectDefaultBenchmarks({ ...data, runs: [] }, filters)).toEqual([]);
  expect(selectors.selectDefaultBenchmarks(data, { ...filters, targets: ['gfx1250'] }).map((item) => item.id)).toEqual(['other-target']);
  expect(data).toEqual(before);
});

test('same-name workloads remain independently searchable by their exact IDs', () => {
  expect(filterBenchmarkOptions(catalog, 'measured').map((item) => item.id)).toEqual(['measured']);
});

test('trend defaults to ALL without replacing an explicit range', () => {
  expect(readDashboardRoute('https://example.test/').historyRange).toBe('ALL');
  expect(readDashboardRoute('https://example.test/?range=1W').historyRange).toBe('1W');
});
