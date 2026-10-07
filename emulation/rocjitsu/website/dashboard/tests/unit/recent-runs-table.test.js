import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import RecentRuns from '../../src/components/overview/RecentRuns.jsx';
import * as recent from '../../src/components/overview/overviewPresentation.js';
const source = vi.hoisted(() => ({ limit: null, rows: [] }));
vi.mock('../../src/data/selectors.js', () => ({ selectRecentRuns: (_data, _filters, limit) => { source.limit = limit; return source.rows.slice(0, limit); } }));
const render = () => renderToStaticMarkup(createElement(RecentRuns, { data: {}, filters: {} }));

test('recent canonical runs requests latest twenty and renders six equal sticky columns with distinct status', () => {
  source.rows = Array.from({ length: 25 }, (_, i) => ({ run: { runId: `fictional-${i}`, timestamp: '2026-10-05T14:02:00Z', commitTimestamp: '2026-10-05T13:48:00Z', provenance: { rocjitsuCommitSha: 'f7e2a9'.padEnd(40, '0'), commitMessage: `Full fictional commit message ${i}` } }, total: i === 2 ? 0 : 2, completed: i === 1 ? 1 : i === 2 ? 0 : 2, failed: i === 1 ? 1 : 0, timeout: 0, duration: i === 1 || i === 2 ? null : 42.3 }));
  const html = render();
  expect(source.limit).toBe(20);
  expect(html.match(/data-run-id="/g)).toHaveLength(20);
  const headings = [...html.matchAll(/<th\b[^>]*>(.*?)<\/th>/g)].map((match) => match[1].replace(/<[^>]+>/g, ''));
  expect(headings).toEqual(['Commit', 'Run time (UTC)', 'Commit time (UTC)', 'Coverage', 'Duration', 'Status']);
  expect(html).toContain('✓ OK');
  expect(html).toContain('× Fail');
  expect(html).toContain('Unavailable');
  expect(html).toContain('Full fictional commit message 0');
  expect(html).toContain('Showing 1–5 of 20 runs');
  expect(html).toContain('position:sticky');
  expect(html).toContain('height:344px');
  expect(html).toContain('width:16.6667%');
  expect(html).toContain('tabindex="0"');
});

test('recent scroll range clamps at final rows and shorter lists have no artificial rows', () => {
  expect(recent.recentVisibleRange?.(20, 900)).toEqual({ start: 16, end: 20 });
  expect(recent.recentVisibleRange?.(20, 9999)).toEqual({ start: 16, end: 20 });
  expect(recent.recentVisibleRange?.(2, 0)).toEqual({ start: 1, end: 2 });
  source.rows = [];
  expect(render()).toContain('No runs available');
  source.rows = [{ run: { runId: 'one', timestamp: '2026-10-05T14:02:00Z', commitTimestamp: '2026-10-05T13:48:00Z' }, total: 0, completed: 0 }];
  expect(render().match(/data-run-id="/g)).toHaveLength(1);
});
