import { act, Children, createElement, isValidElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, test, vi } from 'vitest';
import RecentRuns from '../../src/components/overview/RecentRuns.jsx';
import { selectRecentRunAttempts, selectRecentRunSummaries, selectRecentRuns } from '../../src/data/selectors.js';
import { createRecentRunsHistory } from '../fixtures/recent-runs-history.js';

const filters = { targets: ['gfx1250'], suites: ['Triton'], modes: ['ST'] };
const roots = [];
afterEach(() => { for (const root of roots.splice(0)) act(() => root.unmount()); vi.unstubAllGlobals(); });

test('paged summaries preserve legacy ordering and coverage for selected and explicitly empty scopes', () => {
  const { data } = createRecentRunsHistory();
  const original = [...data.runs];
  const attempts = selectRecentRunAttempts(data);
  expect(data.runs).toEqual(original);
  for (const scope of [filters, ...['targets', 'suites', 'modes'].map((key) => ({ ...filters, [key]: [] }))]) {
    const legacy = selectRecentRuns(data, scope, Infinity);
    const rows = [0, 20, 40, 60].flatMap((start) => selectRecentRunSummaries(attempts.slice(start, start + 20), scope));
    expect(rows).toHaveLength(65);
    rows.forEach((row, index) => {
      expect(legacy[index]).toMatchObject(row);
      expect(row).not.toHaveProperty('baseline');
      expect(row).not.toHaveProperty('durationDelta');
    });
    expect(selectRecentRuns(data, scope)).toEqual(legacy.slice(0, 20));
  }
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

// Mount real React hooks while inspecting the returned MUI tree, without a DOM renderer.
function mountRecentRuns(initialData) {
  const browser = new EventTarget(), document = new EventTarget(), container = new EventTarget();
  Object.assign(container, { nodeType: 1, tagName: 'DIV', namespaceURI: 'http://www.w3.org/1999/xhtml', ownerDocument: document });
  Object.assign(document, { nodeType: 9, body: container, activeElement: null, defaultView: browser });
  browser.HTMLElement = class {};
  browser.HTMLIFrameElement = class extends browser.HTMLElement {};
  browser.document = document;
  vi.stubGlobal('window', browser); vi.stubGlobal('document', document); vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  let tree;
  function Probe({ data }) { tree = RecentRuns({ data, filters }); return null; }
  const root = createRoot(container); roots.push(root);
  const render = (data) => act(() => root.render(createElement(Probe, { data })));
  render(initialData);
  return {
    render,
    page: () => nodes(tree).find((node) => node.props['aria-label'] === 'Recent runs pages'),
    rows: () => nodes(tree).filter((node) => node.props['data-run-id']).map((node) => node.props['data-run-id']),
  };
}

test('Recent Runs evaluates only visible summaries once and never scans results for baselines', () => {
  const { data } = createRecentRunsHistory();
  const reads = new Map();
  const trackedData = { ...data, runs: data.runs.map((run) => ({ ...run, get tests() {
    reads.set(run.runId, (reads.get(run.runId) ?? 0) + 1);
    return run.tests;
  } })) };
  const probe = mountRecentRuns(trackedData);
  const expectOnlyVisibleSummaries = () => {
    expect([...reads.keys()].sort()).toEqual([...probe.rows()].sort());
    expect([...reads.values()]).toEqual(Array(probe.rows().length).fill(1));
  };
  expectOnlyVisibleSummaries();
  reads.clear();
  act(() => probe.page().props.onChange(null, 3));
  expect(probe.rows()).toHaveLength(20);
  expect(probe.rows()[0]).toBe('fictional-pagination-024');
  expectOnlyVisibleSummaries();
  reads.clear();
  act(() => probe.page().props.onChange(null, 4));
  expect(probe.rows()).toHaveLength(5);
  expectOnlyVisibleSummaries();
});

test('mounted Recent Runs retains the clamped page when history shrinks then grows', () => {
  const full = createRecentRunsHistory().data;
  const probe = mountRecentRuns(full);
  act(() => probe.page().props.onChange(null, 4));
  expect(probe.page().props.page).toBe(4);
  probe.render(createRecentRunsHistory(24).data);
  expect(probe.page().props.page).toBe(2);
  expect(probe.rows()).toHaveLength(4);
  probe.render(full);
  expect(probe.page().props.page).toBe(2);
  expect(probe.rows()).toHaveLength(20);
  expect(probe.rows()[0]).toBe('fictional-pagination-044');
  probe.render(createRecentRunsHistory(0).data);
  expect(probe.page()).toBeUndefined();
  expect(probe.rows()).toEqual([]);
  probe.render(full);
  expect(probe.page().props.page).toBe(1);
  expect(probe.rows()[0]).toBe('fictional-pagination-064');
});
