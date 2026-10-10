import { expect, test } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { createElement } from 'react';
import BenchmarksView from '../../src/components/views/BenchmarksView.jsx';
import { benchmarkData } from '../fixtures/publishedData.js';
import { selectBenchmarkSeries } from '../../src/data/selectors.js';
import * as model from '../../src/components/benchmarks/benchmarkExplorer.js';

test('grid-first empty page does not resurrect legacy Single/Grid or aggregate modes', () => {
  const html = renderToStaticMarkup(createElement(BenchmarksView, {
    data: { ...benchmarkData, testCatalog: [], runs: [] },
    filters: { targets: [], suites: [], modes: [] }, selectedRunIds: ['selected'],
    onSelectRun: () => {}, onClearSelectedRuns: () => {},
  }));
  expect(html).not.toContain('Aggregate');
  expect(html).toContain('No benchmarks available');
});

test('search bounds rendered options while searching the entire catalog', () => {
  const options = Array.from({ length: 150 }, (_, i) => ({ id: `${i}`, name: `Benchmark ${i}`, suite: 'Suite' }));
  expect(model.filterBenchmarkOptions?.(options, '')?.length).toBe(50);
  expect(model.filterBenchmarkOptions?.(options, 'Benchmark 149')).toEqual([options[149]]);
});

test('grid selections deduplicate, cap at eight, and allow removal and re-addition', () => {
  const options = Array.from({ length: 10 }, (_, i) => ({ id: `${i}` }));
  expect(model.boundedGridSelection?.([...options, options[0]])).toEqual(options.slice(0, 8));
  const remaining = options.slice(0, 8).filter((o) => o.id !== '3');
  expect(model.boundedGridSelection?.([...remaining, options[3]])).toEqual([...remaining, options[3]]);
  expect(model.boundedGridSelection?.([])).toEqual([]);
});

test('accessible choices retain failures, timeouts, unavailable cells and immutable run identity', () => {
  const runs = ['completed', 'failed', 'timeout', 'missing'].map((status, i) => ({ runId: `run-${i}`, timestamp: `2026-01-0${i + 1}T00:00:00Z`, provenance: { rocjitsuCommitSha: 'abcdef123456' }, tests: status === 'missing' ? [] : [{ target: 'gfx950', mode: 'ST', suite: 'A', logicalTestId: 'a', status, durationSeconds: status === 'completed' ? 2 : null }] }));
  const vm = selectBenchmarkSeries({ runs }, { targets: ['gfx950'], suites: ['A'], modes: ['ST'] }, 'a');
  const choices = model.benchmarkResultChoices?.(vm);
  expect(choices?.map((o) => o.status)).toEqual(['unavailable', 'timeout', 'failed', 'completed']);
  expect(choices?.[0].record).toBeNull();
  expect(choices?.every((o) => o.label.includes(o.run.runId))).toBe(true);
  expect(vm.series[0].points.map((p) => p?.value ?? null)).toEqual([2, null, null, null]);
});

test('catalog respects target membership as well as suites without hiding failures', () => {
  const data = { testCatalog: [{ id: 'a', suite: 'A' }, { id: 'b', suite: 'A' }, { id: 'c', suite: 'B' }], runs: [{ tests: [
    { logicalTestId: 'a', target: 'gfx950', mode: 'ST', status: 'failed' },
    { logicalTestId: 'b', target: 'gfx1250', mode: 'ST', status: 'completed' },
    { logicalTestId: 'c', target: 'gfx950', mode: 'ST', status: 'timeout' },
  ] }] };
  expect(model.explorerCatalog?.(data, { targets: ['gfx950'], suites: ['A'], modes: ['ST'] })).toMatchObject({ available: [{ id: 'a' }], hiddenCount: 2 });
  expect(model.explorerCatalog?.(data, { targets: [], suites: ['A'], modes: ['ST'] }).available).toEqual([]);
});
