import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ThemeProvider } from '@mui/material/styles';
import { describe, expect, test } from 'vitest';
import CategoryTag, { CategoryMarker } from '../../src/components/shared/CategoryTag.jsx';
import { createDashboardTheme } from '../../src/theme/theme.js';
import { colorTokens, suiteColors, suiteTextColor } from '../../src/theme/tokens.js';
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

describe.each(['light', 'dark'])('%s suite colors', (mode) => {
  const theme = createDashboardTheme(mode);
  // Inherited object keys must never resolve to Object.prototype members; they fall back to a stable palette color.
  test.each(inheritedNames)('gives validated suite %s a stable palette color and renders its label', (suite) => {
    const label = validatedSuite(suite);
    const color = suiteTextColor(mode, label);
    expect(color).toMatch(/^#[0-9a-f]{6}$/i);
    expect(suiteTextColor(mode, label)).toBe(color);
    expect([...Object.values(suiteColors[mode]), colorTokens[mode].roseText, colorTokens[mode].oliveText]).toContain(color);
    const html = renderToStaticMarkup(createElement(ThemeProvider, { theme }, createElement(CategoryTag, { label })));
    expect(html).toContain(`<span>${label}</span>`);
    expect(CategoryMarker({ label }).props['data-category-shape']).toBe('circle');
  });
});
