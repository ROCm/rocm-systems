import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import RecentRuns from '../../src/components/overview/RecentRuns.jsx';
import { ThemeProvider } from '@mui/material';
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

const render = (Component, props) => renderToStaticMarkup(createElement(Component, props));
