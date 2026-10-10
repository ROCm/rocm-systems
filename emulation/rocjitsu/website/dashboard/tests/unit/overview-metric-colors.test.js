import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import MetricsGrid from '../../src/components/overview/MetricsGrid.jsx';

const render = (metrics) => renderToStaticMarkup(createElement(MetricsGrid, { metrics }));
test('all four Overview icons are neutral gray when no selected results exist', () => {
  const html = render({ duration: null, durationDelta: null, total: 0, completed: 0, failed: 0 });
  expect([...html.matchAll(/data-icon-tone="([^"]+)"/g)].map((match) => match[1])).toEqual(['neutral', 'neutral', 'neutral', 'neutral']);
});
test.each([[0.1, 'error'], [-0.1, 'success'], [0, 'neutral'], [null, 'neutral']])('Perf change %s has %s value and icon tones', (durationDelta, tone) => {
  const html = render({ duration: 120, durationDelta, total: 2, completed: 2, failed: 0 });
  expect(html).toContain(`data-testid="metric-card-perf-change" data-tone="${tone}" data-icon-tone="${tone}"`);
});
