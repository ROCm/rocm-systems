import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ThemeProvider } from '@mui/material/styles';
import { describe, expect, test } from 'vitest';
import CategoryTag, { CategoryMarker } from '../../src/components/shared/CategoryTag.jsx';
import { createDashboardTheme } from '../../src/theme/theme.js';
import { categoryColors, categoryTextColor } from '../../src/theme/tokens.js';
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
  // Inherited object keys must never resolve to Object.prototype members; they fall back to a stable palette color.
  test.each(inheritedNames)('gives validated suite %s a stable palette color and renders its label', (suite) => {
    const label = validatedSuite(suite);
    const color = categoryTextColor(mode, label, theme.palette.text.secondary, 'suite');
    expect(color).toMatch(/^#[0-9a-f]{6}$/i);
    expect(categoryTextColor(mode, label, theme.palette.text.secondary, 'suite')).toBe(color);
    const html = renderToStaticMarkup(createElement(ThemeProvider, { theme }, createElement(CategoryTag, { label })));
    expect(html).toContain(`<span>${label}</span>`);
    expect(CategoryMarker({ label }).props['data-category-shape']).toBe('circle');
  });
});
