import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import BranchRunsView from '../../src/components/views/BranchRunsView.jsx';
import { readDashboardRoute, buildDashboardUrl } from '../../src/hooks/useDashboardState.js';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

function publication({ onlyOther = false } = {}) {
  const source = createSchema2Publication();
  for (const catalog of Object.values(source.catalogs)) {
    const other = Object.entries(catalog.configurations).filter(([key]) => key.startsWith('gfx1250:'));
    if (onlyOther) catalog.configurations = {};
    for (const [key, ids] of other) catalog.configurations[key.replace('gfx1250:', 'gfx1201:')] = ids;
  }
  for (const [index, run] of source.runs.entries()) {
    const other = run.configurations.filter((c) => c.target === 'gfx1250').map((c) => ({ ...structuredClone(c), target: 'gfx1201' }));
    run.configurations = onlyOther ? other : [...run.configurations, ...(index === 0 ? other : [])];
  }
  return validatePublishedDashboardData(source).data;
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
  for (const target of ['gfx1201', 'gfx1250', 'gfx950']) for (const mode of ['ST', 'MT']) {
    expect(html).toContain(`data-testid="branch-config-${target}-${mode}"`);
  }
});

test('a single other architecture defaults to its own published configuration, not gfx1250', () => {
  const html = render(publication({ onlyOther: true }));
  expect(html).toContain('data-testid="branch-config-gfx1201-ST" aria-pressed="true"');
  expect(html).not.toContain('branch-config-gfx1250-');
  expect(html).not.toContain('branch-config-gfx950-');
});

test('explicit missing target identity remains selectable and unavailable alongside published targets', () => {
  const href = 'https://example.test/?view=branch&target=gfx9999&mode=MT';
  const route = readDashboardRoute(href);
  expect(readDashboardRoute(buildDashboardUrl(href, route))).toEqual(route);
  const html = render(publication({ onlyOther: true }), href);
  expect(html).toContain('data-testid="branch-config-gfx9999-MT" aria-pressed="true"');
  expect(html).toContain('Candidate configuration not published');
  expect(html).toContain('data-testid="branch-config-gfx1201-MT"');
});
