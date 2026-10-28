import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import BranchEnvironment from '../../src/components/branch/BranchEnvironment.jsx';
import RunMetadataDiff from '../../src/components/compare/RunMetadataDiff.jsx';

const run = {
  runId: 'fictional', tests: [], configurations: [], environment: [],
  provenance: { details: ['compiler.version', 'runtime.version', 'sdk'].map((field) => ({
    key: `auxiliary.${field}`, label: `auxiliary.${field}`, value: field,
  })) },
};

test.each(['__proto__', 'constructor', 'toString', 'hasOwnProperty'])('metadata namespace %s remains a literal, renderable heading', (namespace) => {
  const value = { ...run, provenance: { details: [{ key: `${namespace}.version`, label: `${namespace}.version`, value: 'literal version' }] } };
  for (const Component of [BranchEnvironment, RunMetadataDiff]) {
    const html = renderToStaticMarkup(createElement(Component, {
      candidate: value, reference: value, baseline: value, selection: { target: 'gfx950', mode: 'single' },
      filters: { targets: [], suites: [], modes: [] },
    }));
    expect(html).toContain(`>${namespace}</th>`);
    expect(html).toContain('literal version');
  }
});

test.each([BranchEnvironment, RunMetadataDiff])('%s retains relative metadata paths within a namespace', (Component) => {
  const html = renderToStaticMarkup(createElement(Component, {
    candidate: run, reference: run, baseline: run, selection: { target: 'gfx950', mode: 'single' },
    filters: { targets: [], suites: [], modes: [] },
  }));
  expect(html).toContain('>compiler.version</th>');
  expect(html).toContain('>runtime.version</th>');
  expect(html).toContain('>sdk</th>');
});
