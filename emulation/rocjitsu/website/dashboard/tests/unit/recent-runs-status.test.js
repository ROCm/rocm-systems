import { renderToStaticMarkup } from 'react-dom/server';
import { themed } from '../helpers/themed.js';
import { expect, test } from 'vitest';
import raw from '../fixtures/fictional-producer-run.json' with { type: 'json' };
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { selectRecentRunAttempts } from '../../src/data/selectors.js';
import RecentRuns from '../../src/components/overview/RecentRuns.jsx';
import { createFeedbackPublication } from '../fixtures/feedback-publication.js';

test('interrupted fixture displays its run time and ordinary Fail status', () => {
  const { data } = adaptPublishedDashboardData(createFeedbackPublication());
  const run = data.runs.find((run) => run.runId === 'fictional-unfinished-develop');
  const html = renderToStaticMarkup(themed(RecentRuns, {
    data: { ...data, runs: [run] },
    filters: { targets: data.targets, suites: data.suites, modes: data.modes },
  }));
  expect(html).toContain(`dateTime="${run.timestamp}"`);
  expect(html).toMatch(/data-tone="error"[^>]*>× Fail<\/span>/);
});

test('a run with no finish record is dated and ordered by its start timestamp, not treated as the oldest', () => {
  const publication = createFeedbackPublication();
  const interrupted = publication.runs.find((run) => run.publication.id === 'fictional-unfinished-develop');
  const newest = Math.max(...publication.runs.map((run) => Date.parse(run.execution_summary.timestamp)));
  interrupted.execution_summary.timestamp = new Date(newest + 60_000).toISOString();
  expect(interrupted.execution_summary.finished_at).toBeNull();
  const { data } = adaptPublishedDashboardData(publication);
  expect(data.runs.find((run) => run.runId === 'fictional-unfinished-develop').timestamp).toBe(interrupted.execution_summary.timestamp);
  expect(selectRecentRunAttempts(data)[0].runId).toBe('fictional-unfinished-develop');
});

test('a completed run is ordered by its start, not its finish, even when it finished after a later run started', () => {
  const publication = createFeedbackPublication();
  const [early, late] = publication.runs.filter((run) => run.publication.source.branch === 'develop');
  const start = Math.max(...publication.runs.map((run) => Date.parse(run.execution_summary.timestamp)));
  // early starts first but finishes last; late starts second and finishes first
  Object.assign(early.execution_summary, { timestamp: new Date(start + 60_000).toISOString(), finished_at: new Date(start + 600_000).toISOString() });
  Object.assign(late.execution_summary, { timestamp: new Date(start + 120_000).toISOString(), finished_at: new Date(start + 180_000).toISOString() });
  const { data } = adaptPublishedDashboardData(publication);
  const [first, second] = selectRecentRunAttempts(data);
  expect(first.runId).toBe(late.publication.id);
  expect(second.runId).toBe(early.publication.id);
  expect(first.timestamp).toBe(late.execution_summary.timestamp);
});

function renderStatus(finishedAt, status) {  // finishedAt: the run's finish record (null = killed harness); the start timestamp stays valid
  const run = structuredClone(raw);
  run.publication.id = 'fictional-status';
  run.publication.source.branch = 'develop';
  run.execution_summary.timestamp = '2026-10-22T10:00:00Z';
  run.execution_summary.finished_at = finishedAt;
  for (const result of run.benchmark_results) {
    result.status = status;
    result.timing_results_s = status === 'completed' ? [0.25] : [];
    result.exit_code = status === 'completed' ? 0 : null;
    result.error = status === 'completed' ? null : 'Fictional unavailable result';
  }
  const before = structuredClone(run);
  const { data } = adaptPublishedDashboardData({
    index: { generatedAt: '2026-10-22T12:00:00Z', runFiles: ['runs/default-branch/fictional-status.json'] },
    runs: [run],
  });
  const html = renderToStaticMarkup(themed(RecentRuns, {
    data, filters: { targets: data.targets, suites: data.suites, modes: data.modes },
  }));
  expect(run).toEqual(before);
  const match = html.match(/data-tone="([^"]+)"[^>]*>([^<]+)<\/span>/);
  expect(match).not.toBeNull();
  return { tone: match[1], label: match[2] };
}

test.each([
  { status: 'completed', tone: 'success', label: '✓ OK' },
  { status: 'failed', tone: 'error', label: '× Fail' },
  { status: 'timeout', tone: 'error', label: '× Fail' },
])('run without a finish time retains the existing $status-derived status', ({ status, tone, label }) => {
  expect(renderStatus(null, status)).toEqual({ tone, label });
});

test.each([
  { status: 'completed', tone: 'success', label: '✓ OK' },
  { status: 'failed', tone: 'error', label: '× Fail' },
  { status: 'timeout', tone: 'error', label: '× Fail' },
])('finished run retains $status status', ({ status, tone, label }) => {
  expect(renderStatus('2026-10-22T11:00:00Z', status)).toEqual({ tone, label });
});
