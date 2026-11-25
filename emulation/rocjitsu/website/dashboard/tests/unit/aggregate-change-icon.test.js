import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import ComparisonMetrics from '../../src/components/compare/ComparisonMetrics.jsx';
import { formatPercent } from '../../src/utils/formatters.js';

it.each([0, -2, 2, null])('aggregate change has no decorative dash for %s but preserves its formatted value', (delta) => {
  const html = renderToStaticMarkup(createElement(ComparisonMetrics, {
    model: { comparable: [], notComparable: [], aggregateDelta: delta },
  }));
  expect(html).not.toContain('RemoveRoundedIcon');
  expect(html).toContain(formatPercent(delta));
});
