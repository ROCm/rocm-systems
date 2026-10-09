import { act, Children, createElement, isValidElement, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, expect, it, vi } from 'vitest';
import App from '../../src/App.jsx';
import { useDashboardState } from '../../src/hooks/useDashboardState.js';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

const loader = vi.hoisted(() => ({ requests: [] }));
vi.mock('../../src/data/publishedDataUrls.js', () => ({ resolvePublishedDataUrls: () => ({ indexUrl: 'https://example.test/data/index.json' }) }));
vi.mock('../../src/data/dashboardData.js', async (original) => ({
  ...(await original()),
  loadDashboardDataFiles: (options) => new Promise((resolve) => loader.requests.push({ options, resolve })),
}));
const roots = [];
afterEach(() => {
  for (const root of roots.splice(0)) act(() => root.unmount());
  loader.requests = [];
  vi.unstubAllGlobals();
});

const options = { targets: ['gfx1250'], suites: ['Triton'], modes: ['ST', 'MT'] };
it.each([
  ['fresh defaults', '', null, ['ST', 'MT']],
  ['saved ST', '', ['ST'], ['ST']],
  ['saved MT', '', ['MT'], ['MT']],
  ['saved empty', '', [], []],
  ['URL overrides saved ST', '?modes=MT', ['ST'], ['MT']],
  ['URL empty overrides saved modes', '?modes=', ['ST', 'MT'], []],
])('v11 global modes preserve %s', (label, query, saved, expected) => {
  vi.stubGlobal('window', {
    location: { href: `https://example.test/${query}` },
    localStorage: { getItem: () => JSON.stringify({ modes: saved }) },
  });
  let state;
  function Probe() { state = useDashboardState(options); return null; }
  renderToStaticMarkup(createElement(Probe));
  expect(state.modes).toEqual(expected);
});

function nodes(tree) {
  const result = [];
  function visit(element) {
    if (!isValidElement(element)) return;
    result.push(element);
    Children.forEach(element.props.children, visit);
  }
  Children.forEach(tree, visit);
  return result;
}

it('v11 full branch comparison preserves the atomic pair and scrolls to the top after rendering', async () => {
  const browser = new EventTarget();
  const document = new EventTarget();
  const container = new EventTarget();
  Object.assign(container, { nodeType: 1, tagName: 'DIV', namespaceURI: 'http://www.w3.org/1999/xhtml', ownerDocument: document });
  Object.assign(document, { nodeType: 9, body: container, activeElement: null, defaultView: browser });
  browser.HTMLElement = class {};
  browser.HTMLIFrameElement = class extends browser.HTMLElement {};
  browser.document = document;
  browser.location = { href: 'https://example.test/?view=branch&campaign=keep' };
  browser.localStorage = { getItem: () => null, setItem: vi.fn() };
  browser.matchMedia = () => ({ matches: false });
  browser.history = { state: null, pushState: vi.fn((state, title, href) => { browser.location.href = href; }), replaceState: vi.fn((state, title, href) => { browser.location.href = href; }) };
  const frames = [];
  browser.requestAnimationFrame = vi.fn((callback) => frames.push(callback));
  browser.scrollTo = vi.fn();
  vi.stubGlobal('window', browser);
  vi.stubGlobal('document', document);
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  let appTree;
  let dashboardTree;
  function CaptureDashboard({ element }) { dashboardTree = element.type(element.props); return null; }
  function CaptureApp() {
    appTree = App();
    const element = nodes(appTree).find((node) => node.type?.name === 'Dashboard');
    return element ? createElement(CaptureDashboard, { element }) : null;
  }
  const root = createRoot(container);
  roots.push(root);
  act(() => root.render(createElement(StrictMode, null, createElement(CaptureApp))));
  const request = loader.requests.findLast(({ options: requestOptions }) => !requestOptions.signal.aborted);
  const data = validatePublishedDashboardData(createSchema2Publication()).data;
  await act(async () => request.resolve({ data, sourceData: {}, cacheGeneration: null }));
  const branch = nodes(dashboardTree).find((node) => node.type?.name === 'BranchRunsView');
  expect(branch).toBeDefined();
  browser.history.pushState.mockClear();
  act(() => branch.props.onOpenComparison({ candidateId: 'fictional-branch-01', baselineId: 'fictional-develop-21', target: 'gfx1250', mode: 'MT' }));
  const state = nodes(appTree).find((node) => node.type?.name === 'DashboardShell').props.state;
  expect(state).toMatchObject({ tab: 'compare', comparisonCandidateId: 'fictional-branch-01', comparisonBaselineId: 'fictional-develop-21', filters: { targets: ['gfx1250'], modes: ['MT'] } });
  expect(browser.history.pushState).toHaveBeenCalledTimes(1);
  expect(new URL(browser.location.href).searchParams.get('campaign')).toBe('keep');
  expect(browser.requestAnimationFrame).toHaveBeenCalledTimes(1);
  expect(browser.scrollTo).not.toHaveBeenCalled();
  act(() => frames.shift()());
  expect(browser.scrollTo).toHaveBeenCalledWith({ top: 0, left: 0, behavior: 'instant' });
});
