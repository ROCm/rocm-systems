import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import OverviewView from '../../src/components/views/OverviewView.jsx';
import { expect, test } from 'vitest';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { commitTimestampFor, backfillRunIds } from '../../src/data/runOrdering.js';
import { selectOverviewProjection } from '../../src/data/overviewProjection.js';
import { selectBenchmarkSeries, selectRecentRunAttempts, selectRunComparison } from '../../src/data/selectors.js';
import { benchmarkRangeData } from '../../src/components/benchmarks/benchmarkExplorer.js';
import { formatFullDate, formatShortDate } from '../../src/utils/formatters.js';

test.each([null, undefined, 'not-a-date'])('missing/unusable commit date %j makes only dependent chronology unavailable', (date) => {
  const source = createMaintainedPublication();
  source.runs[0].provenance.rocjitsu.rocjitsu_commit_timestamp = date;
  const { data, sourceData } = adaptPublishedDashboardData(source);
  const run = data.runs.find(({ runId }) => runId === 'fictional-develop-00');
  const filters = { targets: data.targets, suites: data.suites, modes: data.modes };
  expect(commitTimestampFor(run)).toBe(date);
  expect(backfillRunIds(data.runs).has(run.runId)).toBe(false);
  expect(selectRecentRunAttempts(data)).toContain(run);
  expect(selectRunComparison(run, run, filters).comparable.length).toBeGreaterThan(0);
  expect(selectOverviewProjection(data, filters).slots.every((slot) => slot.run !== run)).toBe(true);
  expect(selectBenchmarkSeries(data, filters, 'a').runs).not.toContain(run);
  expect(benchmarkRangeData(data, '1W').runs.length).toBeGreaterThan(0);
  expect(formatFullDate(date)).toBe('-');
  expect(formatShortDate(date)).toBe('-');
  expect(sourceData.runs[0].provenance.rocjitsu.rocjitsu_commit_timestamp).toBe(date);
});

test('an entirely undated history does not invent a commit window from execution or publication', () => {
  const source = createMaintainedPublication();
  for (const run of source.runs) run.provenance.rocjitsu.rocjitsu_commit_timestamp = null;
  const { data } = adaptPublishedDashboardData(source);
  const filters = { targets: data.targets, suites: data.suites, modes: data.modes };
  expect(selectOverviewProjection(data, filters).slots).toEqual([]);
  expect(data.latestCommitRun).toBeNull();
  expect(data.latestRun).not.toBeNull();
  expect(data.allRuns).toHaveLength(44);
});

test('undated runs keep benchmark-set membership available outside the chronology', () => {
  const source = createMaintainedPublication();
  for (const run of source.runs) run.provenance.rocjitsu.rocjitsu_commit_timestamp = null;
  const { data } = adaptPublishedDashboardData(source);
  const filters = { targets: data.targets, suites: data.suites, modes: data.modes };
  const html = renderToStaticMarkup(createElement(OverviewView, { data, viewModel: selectOverviewProjection(data, filters), state: { filters } }));
  expect(html).toContain('fictional-old');
  expect(html).toContain('fictional-current');
});
