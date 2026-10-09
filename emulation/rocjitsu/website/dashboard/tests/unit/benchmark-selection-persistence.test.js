import { fixtureRunPath } from '../fixtures/runPath.js';
import { act, Children, createElement, isValidElement, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, expect, it, vi } from 'vitest';
import App from '../../src/App.jsx';
import BenchmarksView from '../../src/components/views/BenchmarksView.jsx';
import { BenchmarkGridPicker } from '../../src/components/benchmarks/BenchmarkPicker.jsx';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

const loader = vi.hoisted(() => ({ requests: [] }));
vi.mock('../../src/data/publishedDataUrls.js', () => ({ resolvePublishedDataUrls: () => ({ indexUrl: 'https://example.test/data/index.json' }) }));
vi.mock('../../src/data/dashboardData.js', async (original) => ({
  ...(await original()),
  loadDashboardDataFiles: (options) => new Promise((resolve, reject) => loader.requests.push({ options, resolve, reject })),
}));

function nodes(node) {
  const found = [];
  function visit(element) {
    if (!isValidElement(element)) return;
    found.push(element);
    Children.forEach(element.props.children, visit);
    Children.forEach(element.props.action, visit);
  }
  Children.forEach(node, visit);
  return found;
}

const roots = [];
afterEach(() => {
  for (const root of roots.splice(0)) act(() => root.unmount());
  loader.requests = [];
  vi.unstubAllGlobals();
});

// Real App, Dashboard and Benchmarks hooks/callbacks; capture their React trees
// instead of mounting MUI/ECharts host DOM. Loader is the only data boundary mock.
function mountApp(storage = new Map(), blockedStorage = false) {
  const browser = new EventTarget();
  const document = new EventTarget();
  const container = new EventTarget();
  Object.assign(container, { nodeType: 1, tagName: 'DIV', namespaceURI: 'http://www.w3.org/1999/xhtml', ownerDocument: document });
  Object.assign(document, { nodeType: 9, body: container, activeElement: null, defaultView: browser });
  browser.HTMLElement = class {};
  browser.HTMLIFrameElement = class extends browser.HTMLElement {};
  browser.document = document;
  browser.location = { href: 'https://example.test/app/?view=benchmarks&campaign=keep#anchor' };
  if (blockedStorage) Object.defineProperty(browser, 'localStorage', { get() { throw new Error('Storage denied'); } });
  else browser.localStorage = { getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) };
  browser.matchMedia = () => ({ matches: false });
  browser.history = { state: { external: 'retained' }, pushState: vi.fn((state, title, href) => { browser.location.href = href; }), replaceState: vi.fn((state, title, href) => { browser.location.href = href; }) };
  vi.stubGlobal('window', browser);
  vi.stubGlobal('document', document);
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  let appTree;
  let benchmarkTree;
  function CaptureBenchmarks({ element }) { benchmarkTree = BenchmarksView(element.props); return null; }
  function CaptureDashboard({ element }) {
    const tree = element.type(element.props);
    const benchmarks = nodes(tree).find((node) => node.type === BenchmarksView);
    if (!benchmarks) benchmarkTree = null;
    return benchmarks ? createElement(CaptureBenchmarks, { element: benchmarks }) : null;
  }
  function CaptureApp() {
    appTree = App();
    const dashboard = nodes(appTree).find((node) => node.type?.name === 'Dashboard');
    if (!dashboard) benchmarkTree = null;
    return dashboard ? createElement(CaptureDashboard, { element: dashboard }) : null;
  }
  const root = createRoot(container);
  roots.push(root);
  act(() => root.render(createElement(StrictMode, null, createElement(CaptureApp))));
  const shell = () => nodes(appTree).find((node) => node.type?.name === 'DashboardShell');
  const header = () => shell().props.header;
  return {
    browser, storage, root,
    get state() { return shell().props.state; },
    get tree() { return benchmarkTree; },
    cards: () => nodes(benchmarkTree).filter((node) => node.props['data-testid'] === 'benchmark-grid-card'),
    ids: () => nodes(benchmarkTree).filter((node) => node.props['data-testid'] === 'benchmark-grid-card').map((node) => node.key),
    click: (label) => act(() => nodes(benchmarkTree).find((node) => node.props['aria-label'] === label).props.onClick()),
    navigate: (tab) => act(() => shell().props.state.setTab(tab)),
    reload: () => act(() => header().props.onReloadData()),
    settle: async (data) => {
      const request = loader.requests.findLast(({ options }) => !options.signal.aborted);
      await act(async () => request.resolve({ data, sourceData: {}, cacheGeneration: null }));
    },
    apply: (selection) => {
      act(() => nodes(benchmarkTree).find((node) => node.props.children === 'Add benchmarks').props.onClick());
      act(() => nodes(benchmarkTree).find((node) => node.type === BenchmarkGridPicker).props.onChange(selection));
      act(() => nodes(benchmarkTree).find((node) => node.props.children === 'Apply selection').props.onClick());
    },
  };
}

it.each(['empty', 'unavailable'])('UI-004 restores %s full workload selection on a fresh App root', async (choice) => {
  const data = validatePublishedDashboardData(createSchema2Publication()).data;
  const selected = choice === 'empty' ? [] : [{ id: `fictional-exact-${'z'.repeat(240)}`, name: 'Fictional retained name', suite: 'Triton', problem: { dataType: false, size: 16 } }];
  const first = mountApp();
  await first.settle(data);
  first.apply(selected);
  expect(first.ids()).toEqual(selected.map((test) => test.id));
  act(() => first.root.unmount());
  roots.splice(roots.indexOf(first.root), 1);
  const restored = mountApp(first.storage);
  // Bootstrap must retain the choice before any replacement data arrives.
  expect(restored.state.selectedBenchmarks).toEqual(selected);
  await restored.settle(data);
  expect(restored.ids()).toEqual(selected.map((test) => test.id));
  expect(restored.state.selectedBenchmarks).toEqual(selected);
});

it.each(['removed', 'empty', 'unavailable'])('UI-004 retains %s workload choice through real App navigation and data reload', async (choice) => {
  const data = validatePublishedDashboardData(createSchema2Publication()).data;
  const before = structuredClone(data);
  const probe = mountApp();
  await probe.settle(data);
  expect(probe.ids()).toEqual(['a', 'b', 'c', 'd']);
  let expected;
  let nextData = data;
  if (choice === 'removed') {
    probe.click('Remove Fictional GEMM from grid');
    expected = ['b', 'c', 'd'];
  } else if (choice === 'empty') {
    for (const id of ['a', 'b', 'c', 'd']) probe.click(`Remove ${data.testCatalog.find((test) => test.id === id).name} from grid`);
    expected = [];
  } else {
    const retained = data.testCatalog.find((test) => test.id === 'c');
    probe.apply([retained]);
    expected = [retained.id];
    const publication = createSchema2Publication();
    publication.runs = publication.runs.filter((run) => run.testCatalog !== 'test-catalogs/fictional-old.json');
    publication.index.runFiles = publication.runs.map(fixtureRunPath);
    nextData = validatePublishedDashboardData(publication).data;
    expect(nextData.testCatalog.some((test) => test.id === retained.id)).toBe(false);
  }
  expect(probe.ids()).toEqual(expected);
  probe.navigate('overview');
  probe.navigate('benchmarks');
  expect(probe.ids()).toEqual(expected);
  probe.reload();
  expect(probe.tree).toBeNull();
  await probe.settle(nextData);
  expect(probe.ids()).toEqual(expected);
  expect(probe.state.selectedBenchmarks.map((test) => test.id)).toEqual(expected);
  if (choice === 'unavailable') {
    expect(nodes(probe.tree).some((node) => node.props.children === 'Fictional retired workload')).toBe(true);
    expect(nodes(probe.tree).some((node) => node.props.children === 'Not in this configuration')).toBe(true);
    expect(probe.state.selectedBenchmarks[0]).toEqual(data.testCatalog.find((test) => test.id === 'c'));
  }
  expect(data).toEqual(before);
});

it('UI-004 derives first-four defaults only while unchosen, without history or workload preference writes', async () => {
  const data = validatePublishedDashboardData(createSchema2Publication()).data;
  const probe = mountApp();
  const replacements = probe.browser.history.replaceState.mock.calls.length;
  expect(probe.state.selectedBenchmarks).toBeNull();
  await probe.settle(data);
  expect(probe.ids()).toEqual(['a', 'b', 'c', 'd']);
  expect(probe.state.selectedBenchmarks).toBeNull();
  expect(probe.storage.has('rocjitsu-dashboard-benchmarks')).toBe(false);
  expect(probe.browser.history.pushState).not.toHaveBeenCalled();
  expect(probe.browser.history.replaceState).toHaveBeenCalledTimes(replacements);
  probe.apply([]);
  expect(probe.state.selectedBenchmarks).toEqual([]);
  expect(JSON.parse(probe.storage.get('rocjitsu-dashboard-benchmarks'))).toEqual([]);
  expect(probe.browser.history.pushState).not.toHaveBeenCalled();
  expect(probe.browser.history.replaceState).toHaveBeenCalledTimes(replacements);
});

it('UI-004 retains in-memory choices when optional storage is denied', async () => {
  const data = validatePublishedDashboardData(createSchema2Publication()).data;
  const probe = mountApp(new Map(), true);
  await probe.settle(data);
  probe.apply([]);
  probe.navigate('overview');
  probe.navigate('benchmarks');
  probe.reload();
  await probe.settle(data);
  expect(probe.ids()).toEqual([]);
  expect(probe.state.selectedBenchmarks).toEqual([]);
});

it.each(['{broken', '{}', '[null]', '[{"id":"missing-name"}]', '[{"id":"a","name":"A","suite":"S","problem":{"bad":[]}}]'])('UI-004 treats malformed stored workload definitions as unchosen: %s', (value) => {
  const probe = mountApp(new Map([['rocjitsu-dashboard-benchmarks', value]]));
  expect(probe.state.selectedBenchmarks).toBeNull();
});

it('UI-004 preserves standalone initialBenchmarkIds and distinguishes controlled null from empty', () => {
  const data = validatePublishedDashboardData(createSchema2Publication()).data;
  const filters = { targets: ['gfx1250'], suites: data.suites, modes: ['ST', 'MT'] };
  const onBenchmarksChange = vi.fn();
  const markup = (props) => renderToStaticMarkup(createElement(BenchmarksView, { data, filters, onBenchmarksChange, ...props }));
  const initialEmpty = markup({ initialBenchmarkIds: [] });
  expect(initialEmpty).not.toContain('data-testid="benchmark-grid-card"');
  const initialMissing = markup({ initialBenchmarkIds: ['exact-missing-id'] });
  expect(initialMissing).toContain('exact-missing-id');
  expect(initialMissing).toContain('Not in this configuration');
  expect(markup({ selectedBenchmarks: [], initialBenchmarkIds: ['a'] })).not.toContain('data-testid="benchmark-grid-card"');
  expect(markup({ selectedBenchmarks: null }).match(/data-testid="benchmark-grid-card"/g)).toHaveLength(4);
  expect(onBenchmarksChange).not.toHaveBeenCalled();
});
