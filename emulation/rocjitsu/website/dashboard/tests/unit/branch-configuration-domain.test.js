import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import BranchRunsView from '../../src/components/views/BranchRunsView.jsx';
import { readDashboardRoute, buildDashboardUrl } from '../../src/hooks/useDashboardState.js';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';

function publication({ onlyOther = false } = {}) {
  const source = createMaintainedPublication();
  for (const run of source.runs) {
    const other = run.benchmark_results.filter((c) => c.target === 'gfx1250').map((c) => ({ ...structuredClone(c), target: 'gfx1201' }));
    run.benchmark_results = onlyOther ? other : [...run.benchmark_results, ...other];
  }
  return adaptPublishedDashboardData(source).data;
}

function render(data, href = 'https://example.test/?view=branch') {
  const route = readDashboardRoute(href);
  return renderToStaticMarkup(createElement(BranchRunsView, {
    data, state: { ...route, setBranchSelection() { throw new Error('Rendering must not write route defaults'); } },
    onOpenComparison() {},
  }));
}

test('entering Branch Runs shows every published target, including a third architecture', () => {
  const html = render(publication());
  for (const target of ['gfx1201', 'gfx1250', 'gfx950']) for (const mode of ['single', 'default']) {
    expect(html).toContain(`data-testid="branch-config-${target}-${mode}"`);
  }
});

test('a single other architecture defaults to its own published configuration, not gfx1250', () => {
  const html = render(publication({ onlyOther: true }));
  expect(html).toContain('data-testid="branch-config-gfx1201-single" aria-pressed="true"');
  expect(html).not.toContain('branch-config-gfx1250-');
  expect(html).not.toContain('branch-config-gfx950-');
});

test('explicit missing target identity remains selectable and unavailable alongside published targets', () => {
  const href = 'https://example.test/?view=branch&target=gfx9999&mode=default';
  const route = readDashboardRoute(href);
  expect(readDashboardRoute(buildDashboardUrl(href, route))).toEqual(route);
  const html = render(publication({ onlyOther: true }), href);
  expect(html).toContain('data-testid="branch-config-gfx9999-default" aria-pressed="true"');
  expect(html).toContain('Candidate configuration not published');
  expect(html).toContain('data-testid="branch-config-gfx1201-default"');
});
