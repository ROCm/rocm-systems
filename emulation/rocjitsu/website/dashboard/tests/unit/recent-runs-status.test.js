import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import raw from '../fixtures/fictional-producer-run.json' with { type: 'json' };
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import RecentRuns from '../../src/components/overview/RecentRuns.jsx';
import { createFeedbackPublication } from '../fixtures/feedback-publication.js';

test('interrupted fixture displays its run time and ordinary Fail status', () => {
  const { data } = adaptPublishedDashboardData(createFeedbackPublication());
  const run = data.runs.find((run) => run.runId === 'fictional-unfinished-develop');
  const html = renderToStaticMarkup(createElement(RecentRuns, {
    data: { ...data, runs: [run] },
    filters: { targets: data.targets, suites: data.suites, modes: data.modes },
  }));
  expect(html).toContain(`dateTime="${run.timestamp}"`);
  expect(html).toMatch(/data-tone="error"[^>]*>× Fail<\/span>/);
});

function renderStatus(finishedAt, status) {
  const run = structuredClone(raw);
  run.publication.id = 'fictional-status';
  run.publication.source.branch = 'develop';
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
  const html = renderToStaticMarkup(createElement(RecentRuns, {
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
