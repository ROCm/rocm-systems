import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import DurationHistory from '../../src/components/overview/DurationHistory.jsx';
import OverviewView from '../../src/components/overview/OverviewView.jsx';
import RecentRuns from '../../src/components/overview/RecentRuns.jsx';
import { ThemeProvider, alpha } from '@mui/material';
import DashboardShell from '../../src/components/layout/DashboardShell.jsx';
import FiltersBar from '../../src/components/layout/FiltersBar.jsx';
test.each(['light', 'dark'])('main sidebar has explicit subtle vertical border and stronger section rules in %s', (mode) => {
  const theme = createDashboardTheme(mode);
  theme.components.MuiUseMediaQuery = { defaultProps: { ssrMatchMedia: () => ({ matches: true }) } };
  const state = { tab: 'overview', targets: [], suites: [], modes: [], setTab() {}, setTargets() {}, setSuites() {}, setModes() {} };
  const data = { runs: [{}], targets: [], suites: [], modes: [] };
  const html = render(ThemeProvider, { theme, children: createElement(DashboardShell, { data, state }) });
  expect(html.includes(`border-right:2px solid ${theme.palette.divider}`)).toBe(true);
  expect(html.includes(`border-bottom:1px solid ${alpha(theme.palette.text.primary, 0.3)}`)).toBe(true);
  const filters = render(ThemeProvider, { theme, children: createElement(FiltersBar, { data, state }) });
  expect(filters.includes(`border-top:1px solid ${alpha(theme.palette.text.primary, 0.3)}`)).toBe(true);
  const branch = render(ThemeProvider, { theme, children: createElement(DashboardShell, { data, state: { ...state, tab: 'branch' } }) });
  expect(branch.includes(`border-top:1px solid ${alpha(theme.palette.text.primary, 0.3)}`)).toBe(true);
});
import { createDashboardTheme } from '../../src/theme/theme.js';
const recent = vi.hoisted(() => ({ rows: [] }));
vi.mock('../../src/data/selectors.js', async (importOriginal) => ({
  ...await importOriginal(),
  selectRecentRunAttempts: () => recent.rows.map(({ run }) => run),
  selectRecentRunSummaries: (runs) => runs.map((run) => recent.rows.find((row) => row.run === run)),
}));
const sha = 'AbCdEf0123456789'.padEnd(40, '0');
function renderRecent(source, repository, mode = 'light') {
  recent.rows = [{ run: { runId: 'attempt', source, provenance: { rocjitsuCommitSha: sha } }, total: 1, completed: 1 }];
  return render(ThemeProvider, { theme: createDashboardTheme(mode), children: createElement(RecentRuns, { data: { repository }, filters: {} }) });
}
test.each(['light', 'dark'])('recent commits link exact source SHA and repository with safe blue links in %s', (mode) => {
  const html = renderRecent({ repository: 'https://github.com/example/source', commit: sha }, 'https://github.com/other/top', mode);
  const anchor = html.match(/<a\b[^>]*href="[^"]*\/commit\/[^>]*>[\s\S]*?<\/a>/)?.[0];
  expect(Boolean(anchor)).toBe(true);
  expect(anchor).toContain(`href="https://github.com/example/source/commit/${sha}"`);
  expect(anchor).toContain('target="_blank"');
  expect(anchor).toContain('rel="noopener noreferrer"');
  expect(html.includes(`color:${createDashboardTheme(mode).palette.primary.main}`)).toBe(true);
});
test('legacy commit links use publication repository only when source repository is absent', () => {
  expect(renderRecent(undefined, 'https://github.com/example/legacy').includes(`/commit/${sha}`)).toBe(true);
});
test.each(['javascript:alert(1)', 'https://user:pass@example.test/repo', 'not a URL'])('invalid explicit source repository is never replaced by top-level repository: %s', (repository) => {
  expect(renderRecent({ repository, commit: sha }, 'https://github.com/example/top').includes('/commit/')).toBe(false);
});
test('source commit identity is used for link text even when legacy provenance differs', () => {
  const sourceSha = '1234567890abcdef'.padEnd(40, '1');
  const html = renderRecent({ repository: 'https://github.com/example/source', commit: sourceSha }, undefined);
  expect(html.includes(`>${sourceSha.slice(0, 8)}</a>`)).toBe(true);
});
test('missing repository and malformed SHA render non-link commit identity', () => {
  expect(renderRecent(undefined, undefined).includes('/commit/')).toBe(false);
  expect(renderRecent({ repository: 'https://github.com/example/source', commit: 'bad/sha' }, undefined).includes('/commit/')).toBe(false);
});

const note = 'Trend values are normalized to the latest test catalog using fixed first-success anchors for added benchmarks. Explore original per-benchmark measurements:';
test.each(['1W', '1M', '3M', 'ALL'])('overview always exposes normalization explanation and benchmark action for %s', (range) => {
  const onOpenBenchmarks = vi.fn();
  const html = render(OverviewView, { data: { runs: [] }, viewModel: { history: { ...history, normalized: false }, metrics: { total: 0, estimatedBaseline: false }, changes: [] }, state: { filters: { targets: [], suites: [], modes: [] }, historyRange: range, setHistoryRange() {} }, onOpenBenchmarks });
  expect(html).toContain(note);
  expect(html).toMatch(/<button[^>]*type="button"[^>]*>Benchmarks<\/button>/);
});


const capture = vi.hoisted(() => ({ props: null }));
vi.mock('../../src/components/shared/Chart.jsx', () => ({ default: (props) => { capture.props = props; return null; } }));
const history = {
  series: [{ data: [120, null, 0, 100], baseline: 120 }],
  slots: [0, 1, 2, 3].map((x) => ({ x, run: { runId: `run-${x}`, timestamp: '2026-10-01T12:00:00Z' } })),
  dayKeys: [], axisMax: 3, currentDuration: 100, durationDelta: -16.67,
};
const render = (Component, props) => renderToStaticMarkup(createElement(Component, props));

test('trend connects measured points visually without replacing null or measured-zero slots', () => {
  const before = structuredClone(history);
  render(DurationHistory, { history, range: 'ALL', onRangeChange() {} });
  expect(capture.props.option.series).toHaveLength(2);
  expect(capture.props.option.series[0].connectNulls).toBe(false);
  expect(capture.props.option.series[0].data).toEqual([[0, 2], [1, null], [2, 0], [3, 100 / 60]]);
  expect(capture.props.option.tooltip.formatter([{ dataIndex: 1, value: [1, null] }])).toBe('');
  expect(history).toEqual(before);
});
