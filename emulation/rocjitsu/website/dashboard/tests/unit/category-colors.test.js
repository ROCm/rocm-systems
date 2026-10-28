import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { alpha, ThemeProvider } from '@mui/material/styles';
import { describe, expect, test } from 'vitest';
import CategoryTag, { CategoryMarker } from '../../src/components/shared/CategoryTag.jsx';
import { createDashboardTheme } from '../../src/theme/theme.js';
import { categoryColors } from '../../src/theme/tokens.js';
import { configurationColor, targetColor } from '../../src/utils/chartColors.js';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';

const inheritedNames = ['constructor', 'toString', '__proto__', 'hasOwnProperty'];

function validatedSuite(suite) {
  const publication = createMaintainedPublication();
  for (const run of publication.runs) for (const row of run.benchmark_results) if (row.suite === 'Triton') row.suite = suite;
  const { data } = adaptPublishedDashboardData(publication);
  const label = data.testCatalog.find(({ id }) => id === 'a').suite;
  expect(label).toBe(suite);
  return label;
}

describe.each(['light', 'dark'])('%s category colors', (mode) => {
  const theme = createDashboardTheme(mode);
  test.each(['gfx1250', 'gfx950'])('aligns target and single-policy chart colors with the %s tag', (target) => {
    expect(targetColor(target, 0, mode)).toBe(categoryColors[mode][target]);
    expect(configurationColor({ target, mode: 'single' }, 0, mode)).toBe(categoryColors[mode][target]);
  });
  test.each(inheritedNames)('uses neutral tag and marker colors for validated suite %s', (suite) => {
    const label = validatedSuite(suite);
    const tagStyle = CategoryTag({ label }).props.sx[0];
    expect(tagStyle.bgcolor(theme)).toBe(mode === 'dark' ? theme.palette.dashboard.subtleSurface : alpha(theme.palette.text.secondary, 0.07));
    expect(tagStyle.color(theme)).toBe(theme.palette.text.secondary);
    expect(CategoryMarker({ label }).props.sx[0].bgcolor(theme)).toBe(theme.palette.text.secondary);
    const html = renderToStaticMarkup(createElement(ThemeProvider, { theme }, createElement(CategoryTag, { label })));
    expect(html).toContain(`<span>${label}</span>`);
    expect(html).toContain('data-category-shape="circle"');
  });

  test.each(Object.keys(categoryColors.light))('preserves the recognized color for %s', (label) => {
    const color = categoryColors[mode][label];
    const tagStyle = CategoryTag({ label }).props.sx[0];
    expect(tagStyle.color(theme)).toBe(color);
    expect(tagStyle.bgcolor(theme)).toBe(mode === 'dark' ? theme.palette.dashboard.subtleSurface : alpha(color, 0.07));
    expect(CategoryMarker({ label }).props.sx[0].bgcolor(theme)).toBe(color);
  });
});
