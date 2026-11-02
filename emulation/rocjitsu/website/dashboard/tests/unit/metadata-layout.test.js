import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import BranchEnvironment from '../../src/components/branch/BranchEnvironment.jsx';
import RunMetadataDiff from '../../src/components/compare/RunMetadataDiff.jsx';

const run = {
  runId: 'fictional-attempt', tests: [],
  environmentId: '[["machine.cpu","fictional-cpu"]]',
  configurations: [{ target: 'gfx950', thread_policy: 'single' }, { target: 'gfx950', thread_policy: 'default' }],
  environment: [{ key: 'machine.cpu', label: 'CPU', value: 'fictional-cpu' }],
  provenance: { details: [{ key: 'execution.samples', label: 'Samples', value: 0 }, { key: 'auxiliary.enabled', label: 'Enabled', value: false }] },
};

test('branch metadata presents grouped facts rather than the serialized environment identity', () => {
  const html = renderToStaticMarkup(createElement(BranchEnvironment, { candidate: run, reference: run, selection: { target: 'gfx950', mode: 'single' } }));
  expect(html).not.toContain('Environment identity');
  expect(html).toContain('Machine');
  expect(html).toContain('Execution');
  expect(html).toContain('fictional-cpu');
  expect(html).toContain('>0<');
  expect(html).toContain('>false<');
  expect(html).not.toContain('>execution.samples<');
});

test.each([BranchEnvironment, RunMetadataDiff])('%s displays a namespaced field as sdk without changing its identity', (Component) => {
  const value = { ...run, provenance: { details: [{ key: 'auxiliary.sdk', label: 'auxiliary.sdk', value: 'fictional-sdk' }] } };
  const html = renderToStaticMarkup(createElement(Component, { baseline: value, reference: value, candidate: value, selection: { target: 'gfx950', mode: 'single' }, filters: { targets: [], suites: [], modes: [] } }));
  expect(html).toContain('>sdk</th>');
  expect(html).not.toContain('>auxiliary.sdk<');
  expect(html).toContain('auxiliary.sdk"');
});

test('compared run configurations are readable target and thread-policy entries', () => {
  const html = renderToStaticMarkup(createElement(RunMetadataDiff, { baseline: run, candidate: run, filters: { targets: [], suites: [], modes: [] } }));
  expect(html).not.toContain('&quot;thread_policy&quot;');
  expect(html).toContain('gfx950 · single');
  expect(html).toContain('gfx950 · default');
  expect(html).toContain('Execution');
});
