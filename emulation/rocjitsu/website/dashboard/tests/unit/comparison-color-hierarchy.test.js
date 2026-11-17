import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ThemeProvider } from '@mui/material/styles';
import { expect, test } from 'vitest';
import ComparisonMetrics from '../../src/components/compare/ComparisonMetrics.jsx';
import { createDashboardTheme } from '../../src/theme/theme.js';
import ChangeCountChip from '../../src/components/shared/ChangeCountChip.jsx';

test.each(['light', 'dark'])('%s comparison chips use semantic surfaces with quiet borders', (mode) => {
  const theme = createDashboardTheme(mode);
  const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).join(', ');
  for (const tone of ['success', 'error']) {
    const html = renderToStaticMarkup(createElement(ThemeProvider, { theme }, createElement(ChangeCountChip, { tone, label: '0 results' })));
    // Fill and border are both alpha tints of the text hue, so they cannot disagree.
    const hue = rgb(theme.palette[tone].main);
    expect(html).toContain(`background-color:rgba(${hue}, 0.08)`);
    expect(html).toContain(`border-color:rgba(${hue}, 0.4)`);
  }
  const neutral = renderToStaticMarkup(createElement(ThemeProvider, { theme }, createElement(ChangeCountChip, { label: '0 results' })));
  expect(neutral).toContain(`background-color:${theme.palette.dashboard.subtleSurface}`);
});

function metricColors(mode, delta = 0, comparable = [{}]) {
  const theme = createDashboardTheme(mode);
  const html = renderToStaticMarkup(createElement(ThemeProvider, { theme }, createElement(ComparisonMetrics, {
    candidate: {}, baseline: {},
    model: { comparable, notComparable: [], candidateDuration: 8, baselineDuration: 10, aggregateDelta: delta },
  })));
  const colors = [...html.matchAll(/class="([^"]*MuiTypography-summaryValue[^"]*)"/g)].map(([, classes]) => {
    const cssClass = classes.split(' ').find((name) => name.startsWith('css-'));
    const rule = html.match(new RegExp(`\\.${cssClass}\\{([^}]+)\\}`))[1];
    return rule.match(/(?:^|;)color:([^;]+)/)[1];
  });
  return { colors, theme };
}

test.each(['light', 'dark'])('%s comparison values use primary ink, not label or coverage-status colors', (mode) => {
  const { colors, theme } = metricColors(mode);
  expect(colors).toEqual(Array(5).fill(theme.palette.text.primary));
});

test.each(['light', 'dark'])('%s aggregate preserves outcome colors and unavailable values remain muted', (mode) => {
  const { theme } = metricColors(mode);
  expect(metricColors(mode, -20).colors[0]).toBe(theme.palette.success.main);
  expect(metricColors(mode, 20).colors[0]).toBe(theme.palette.error.main);
  expect(metricColors(mode, null, []).colors.slice(0, 3)).toEqual(Array(3).fill(theme.palette.text.muted));
});
