import { existsSync } from 'node:fs';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import { ThemeProvider } from '@mui/material/styles';
import { createDashboardTheme } from '../../src/theme/theme.js';
import DashboardHeader from '../../src/components/layout/DashboardHeader.jsx';
import DashboardShell from '../../src/components/layout/DashboardShell.jsx';
import { readDashboardRoute } from '../../src/hooks/useDashboardState.js';

test('shared page definitions preserve navigation order, titles and accepted routes', async () => {
  expect(existsSync(new URL('../../src/config/dashboardPages.js', import.meta.url))).toBe(true);
  const { dashboardPages } = await import('../../src/config/dashboardPages.js');
  expect(dashboardPages).toEqual([
    { id: 'overview', label: 'Overview' },
    { id: 'branch', label: 'Branch Runs' },
    { id: 'benchmarks', label: 'Benchmarks' },
    { id: 'compare', label: 'Run Comparison' },
  ]);
  const render = (element) => renderToStaticMarkup(createElement(ThemeProvider, { theme: createDashboardTheme('light') }, element));
  const navigation = render(createElement(DashboardShell, {
    data: {}, state: { tab: 'branch', targets: [], suites: [], modes: [] },
  }));
  let previous = -1;
  for (const { id, label } of dashboardPages) {
    expect(readDashboardRoute(`https://example.test/?view=${id}`)).toMatchObject({ tab: id, routeError: '' });
    const header = render(createElement(DashboardHeader, { tab: id, mode: 'light' }));
    expect(header).toContain(`>${label}</h1>`);
    const position = navigation.indexOf(`id="dashboard-tab-${id}"`);
    expect(position).toBeGreaterThan(previous);
    previous = position;
    const tab = navigation.slice(position, navigation.indexOf('</button>', position));
    expect(tab.replace(/<style\b[^>]*>[\s\S]*?<\/style>/g, '').replace(/<[^>]*>/g, '')).toContain(label);
  }
  expect(readDashboardRoute('https://example.test/?view=removed')).toMatchObject({ tab: 'overview', routeError: 'Invalid view selection in this URL.' });
});
