import { act, Children, createElement, isValidElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, test, vi } from 'vitest';
import BenchmarksView from '../../src/components/views/BenchmarksView.jsx';
import { BenchmarkGridPicker } from '../../src/components/benchmarks/BenchmarkPicker.jsx';
import { explorerCatalog } from '../../src/components/benchmarks/benchmarkExplorer.js';
import { selectBenchmarkCatalog } from '../../src/data/selectors.js';

const catalog = ['st', 'mt', 'absent', 'other-suite', 'other-target'].map((id) => ({
  id, name: `Fictional ${id}`, suite: id === 'other-suite' ? 'Llama' : 'Triton',
}));
const data = { testCatalog: catalog, runs: [{ tests: catalog.filter(({ id }) => id !== 'absent').map((test) => ({
  logicalTestId: test.id, suite: test.suite, target: test.id === 'other-target' ? 'gfx950' : 'gfx1250',
  mode: test.id === 'mt' ? 'MT' : 'ST', status: test.id === 'mt' ? 'failed' : 'completed',
})) }] };
const filters = { targets: ['gfx1250'], suites: ['Triton'] };
const roots = [];

test.each([
  { name: 'shared selector', selectCatalog: selectBenchmarkCatalog },
  { name: 'explorer delegate', selectCatalog: explorerCatalog },
])('$name resolves omitted result suites from the catalog', ({ selectCatalog }) => {
  const input = { ...data, runs: data.runs.map((run) => ({ ...run,
    tests: run.tests.map((test) => { const result = { ...test }; delete result.suite; return result; }),
  })) };
  const before = structuredClone(input);
  for (const [scope, ids] of [
    [filters, ['st', 'mt']],
    [{ ...filters, modes: ['ST'] }, ['st']],
    [{ ...filters, modes: ['MT'] }, ['mt']],
    [{ ...filters, modes: ['ST', 'MT'] }, ['st', 'mt']],
    [{ ...filters, modes: [] }, []],
    [{ ...filters, suites: ['Llama'] }, ['other-suite']],
    [{ ...filters, suites: [] }, []],
    [{ ...filters, targets: ['gfx950'] }, ['other-target']],
    [{ ...filters, targets: [] }, []],
  ]) {
    const result = selectCatalog(input, scope);
    expect(result.all).toBe(catalog);
    expect(result.available).toEqual(catalog.filter(({ id }) => ids.includes(id)));
    expect(result.hiddenCount).toBe(catalog.length - ids.length);
  }
  expect(input).toEqual(before);
});
afterEach(() => { for (const root of roots.splice(0)) act(() => root.unmount()); vi.unstubAllGlobals(); });

function nodes(tree) {
  const result = [];
  function visit(element) {
    if (!isValidElement(element)) return;
    result.push(element);
    Children.forEach(element.props.children, visit);
    Children.forEach(element.props.action, visit);
  }
  Children.forEach(tree, visit);
  return result;
}

// Mount the real view and picker callbacks/hooks; inspect the resulting Autocomplete
// props without mounting MUI's host DOM or ECharts (the browser gate owns those).
function mountPicker() {
  const browser = new EventTarget(), document = new EventTarget(), container = new EventTarget();
  Object.assign(container, { nodeType: 1, tagName: 'DIV', namespaceURI: 'http://www.w3.org/1999/xhtml', ownerDocument: document });
  Object.assign(document, { nodeType: 9, body: container, activeElement: null, defaultView: browser });
  browser.HTMLElement = class {};
  browser.HTMLIFrameElement = class extends browser.HTMLElement {};
  browser.document = document;
  vi.stubGlobal('window', browser); vi.stubGlobal('document', document); vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  let tree, picker;
  function CapturePicker({ element }) { picker = BenchmarkGridPicker(element.props); return null; }
  function Probe({ scope }) {
    tree = BenchmarksView({ data, filters: scope, initialBenchmarkIds: [] });
    return createElement(CapturePicker, { element: nodes(tree).find((node) => node.type === BenchmarkGridPicker) });
  }
  const root = createRoot(container); roots.push(root);
  const render = (scope) => act(() => root.render(createElement(Probe, { scope })));
  render(filters);
  act(() => nodes(tree).find((node) => node.props.children === 'Add benchmarks').props.onClick());
  return {
    render,
    options: () => picker.props.options.map(({ id }) => id),
    choose: (id) => act(() => picker.props.onChange(null, [picker.props.options.find((option) => option.id === id)])),
    apply: () => act(() => nodes(tree).find((node) => node.props.children === 'Apply selection').props.onClick()),
    cards: () => nodes(tree).filter((node) => node.props['data-testid'] === 'benchmark-grid-card').map((node) => node.key),
  };
}

test('mounted benchmark picker defaults omitted modes to ST and MT and keeps explicit empty scopes empty', () => {
  const picker = mountPicker();
  expect(picker.options()).toEqual(['st', 'mt']);
  picker.choose('mt');
  picker.apply();
  expect(picker.cards()).toEqual(['mt']);
  for (const [scope, expected] of [
    [{ ...filters, modes: ['ST'] }, ['st']],
    [{ ...filters, modes: ['MT'] }, ['mt']],
    [{ ...filters, modes: [] }, []],
    [{ ...filters, targets: [] }, []],
    [{ ...filters, suites: [] }, []],
    [filters, ['st', 'mt']],
  ]) {
    picker.render(scope);
    expect(picker.options()).toEqual(expected);
    expect(picker.cards()).toEqual(['mt']);
  }
});
