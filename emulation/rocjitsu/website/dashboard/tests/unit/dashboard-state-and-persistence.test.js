import { createReactRootSurface } from '../helpers/react-root-surface.js';
import { act, createElement, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, expect, it, vi } from 'vitest';
import { useDashboardState } from '../../src/hooks/useDashboardState';

const data = { targets: ['gfx950', 'gfx1250'], suites: ['gemm', 'memory'], modes: ['ST', 'MT'] };
function readState(dataset = data) {
  let state;
  function Probe() { state = useDashboardState(dataset); return null; }
  renderToStaticMarkup(createElement(Probe));
  return state;
}
const roots = [];
afterEach(() => {
  for (const root of roots.splice(0)) act(() => root.unmount());
  vi.unstubAllGlobals();
});

// Probe renders no host elements; only React's root event/selection surface is
// needed. State updates and StrictMode run in real React, without a DOM package.
function mountState({ href = 'https://example.test/app/?campaign=keep#anchor', saved = null, dataset = data, blockedStorage = false } = {}) {
  const { browser, container } = createReactRootSurface();
  browser.location = { href };
  let stored = saved === null ? null : JSON.stringify(saved);
  if (blockedStorage) Object.defineProperty(browser, 'localStorage', { get() { throw new Error('Storage denied'); } });
  else browser.localStorage = { getItem: vi.fn(() => stored), setItem: vi.fn((key, value) => { stored = value; }) };
  const entries = [{ href, state: { external: 'retained' } }];
  let index = 0;
  browser.history = {
    get state() { return entries[index].state; },
    pushState: vi.fn((state, title, nextHref) => {
      entries.splice(index + 1);
      entries.push({ href: nextHref, state });
      index += 1;
      browser.location.href = nextHref;
    }),
    replaceState: vi.fn((state, title, nextHref) => {
      entries[index] = { href: nextHref, state };
      browser.location.href = nextHref;
    }),
  };

  let state;
  let currentData = dataset;
  const renders = [];
  function Probe() { state = useDashboardState(currentData); renders.push(state); return null; }
  const root = createRoot(container);
  roots.push(root);
  const render = () => act(() => root.render(createElement(StrictMode, null, createElement(Probe))));
  render();
  return {
    browser, entries, renders,
    get state() { return state; },
    update: (action) => act(() => action(state)),
    rerender: (nextData) => { currentData = nextData; render(); },
    go: (offset) => act(() => {
      index += offset;
      browser.location.href = entries[index].href;
      browser.dispatchEvent(new Event('popstate'));
    }),
  };
}

it('snapshots global scope on first ordinary comparison visit and then keeps both scopes independent', () => {
  const global = { targets: ['gfx950'], suites: ['memory'], modes: ['ST'] };
  const probe = mountState({ saved: global });
  probe.update((state) => state.setTab('compare'));
  expect(probe.state.filters).toEqual(global);
  probe.update((state) => { state.setTargets(['gfx1250']); state.setModes(['MT']); state.setSuites([]); });
  const comparison = { targets: ['gfx1250'], suites: [], modes: ['MT'] };
  expect(probe.state.filters).toEqual(comparison);
  expect(JSON.parse(probe.browser.localStorage.getItem('rocjitsu-dashboard-filters'))).toEqual(global);
  probe.update((state) => state.setTab('overview'));
  expect(probe.state.filters).toEqual(global);
  probe.update((state) => state.setSuites(['gemm']));
  probe.update((state) => state.setTab('compare'));
  expect(probe.state.filters).toEqual(comparison);
  const href = probe.browser.location.href;
  const reloaded = mountState({ href });
  expect(reloaded.state.filters).toEqual(comparison);
  reloaded.update((state) => state.setTab('benchmarks'));
  expect(reloaded.state.filters).toEqual({ ...global, suites: ['gemm'] });
});

it('scoped comparison opening never edits global filters or their storage', () => {
  const global = { targets: ['gfx950'], suites: ['memory'], modes: ['ST'] };
  const probe = mountState({ saved: global });
  probe.update((state) => state.openComparison({ candidateId: 'c', baselineId: 'b', target: 'gfx1250', mode: 'MT', suites: ['gemm'] }));
  expect(probe.state.filters).toEqual({ targets: ['gfx1250'], suites: ['gemm'], modes: ['MT'] });
  expect(JSON.parse(probe.browser.localStorage.getItem('rocjitsu-dashboard-filters'))).toEqual(global);
  probe.update((state) => state.setTab('benchmarks'));
  expect(probe.state.filters).toEqual(global);
});

it('separate comparison URL scope preserves intentional empties and survives Back/Forward', () => {
  const probe = mountState({ href: 'https://example.test/?view=compare&targets=gfx950&modes=ST&suites=memory&compareTargets=&compareModes=&compareSuites=' });
  expect(probe.state.filters).toEqual({ targets: [], modes: [], suites: [] });
  probe.update((state) => state.setTab('overview'));
  expect(probe.state.filters).toEqual({ targets: ['gfx950'], modes: ['ST'], suites: ['memory'] });
  probe.go(-1);
  expect(probe.state.filters).toEqual({ targets: [], modes: [], suites: [] });
  probe.go(1);
  expect(probe.state.filters.targets).toEqual(['gfx950']);
});

it('resolves comparison defaults after bootstrap and retains its first snapshot across later global changes', () => {
  const probe = mountState({ dataset: { targets: [], suites: [], modes: [] }, href: 'https://example.test/?view=compare' });
  expect(probe.state.filters).toEqual({ targets: [], suites: [], modes: [] });
  probe.rerender(data);
  expect(probe.state.filters).toEqual(data);
  probe.update((state) => state.setTab('overview'));
  probe.update((state) => { state.setTargets([]); state.setModes([]); });
  probe.update((state) => state.setTab('compare'));
  expect(probe.state.filters).toEqual(data);
  probe.rerender({ targets: [], suites: [], modes: [] });
  probe.rerender(data);
  expect(probe.state.filters).toEqual(data);
});

it('does not expose retired benchmark mode or global search state', () => {
  const state = readState();
  for (const key of ['benchmarkMode', 'setBenchmarkMode', 'search', 'setSearch']) {
    expect(state).not.toHaveProperty(key);
  }
});

it('pushes one coherent branch selection even under StrictMode', () => {
  const probe = mountState();
  const selection = { branch: 'feat/a', candidateId: 'missing-attempt', referenceId: 'missing-reference', manual: true, target: 'gfx950', mode: 'MT', detail: true };
  probe.update((state) => state.setBranchSelection(selection));
  expect(probe.state.branchSelection).toEqual(selection);
  expect(probe.browser.history.pushState).toHaveBeenCalledTimes(1);
  const url = new URL(probe.browser.location.href);
  expect(url.searchParams.get('run')).toBe('missing-attempt');
  expect(url.searchParams.get('reference')).toBe('missing-reference');
  expect(url.searchParams.get('campaign')).toBe('keep');
  expect(url.hash).toBe('#anchor');
  expect(probe.browser.history.state).toEqual({ external: 'retained' });
  probe.update((state) => state.setBranchSelection((current) => ({ ...current, detail: false })));
  expect(probe.state.branchSelection).toEqual({ ...selection, detail: false });
  expect(probe.browser.history.pushState).toHaveBeenCalledTimes(2);
});
it('opens comparison with one atomic pair and scope while preserving deliberate empty suites', () => {
  const probe = mountState({ saved: { suites: [], modes: [] } });
  const branch = { branch: 'feat/x', candidateId: 'attempt-x', referenceId: 'develop-x', manual: true, target: 'gfx1250', mode: 'ST', detail: true };
  probe.update((state) => state.setBranchSelection(branch));
  probe.browser.history.pushState.mockClear();
  probe.renders.length = 0;
  probe.update((state) => state.openComparison({ candidateId: 'missing-candidate', baselineId: 'missing-baseline', target: 'gfx950', mode: 'MT' }));
  expect(probe.browser.history.pushState).toHaveBeenCalledTimes(1);
  expect(probe.state).toMatchObject({ tab: 'compare', comparisonCandidateId: 'missing-candidate', comparisonBaselineId: 'missing-baseline', branchSelection: branch });
  expect(probe.state.filters).toEqual({ targets: ['gfx950'], suites: [], modes: ['MT'] });
  for (const rendered of probe.renders) {
    expect(rendered.tab).toBe('compare');
    expect(rendered.comparisonCandidateId).toBe('missing-candidate');
    expect(rendered.comparisonBaselineId).toBe('missing-baseline');
    expect(rendered.filters).toEqual(probe.state.filters);
  }
  const url = new URL(probe.browser.location.href);
  expect(url.searchParams.get('compareCandidate')).toBe('missing-candidate');
  expect(url.searchParams.get('compareBaseline')).toBe('missing-baseline');
  expect(url.searchParams.getAll('compareTargets')).toEqual(['gfx950']);
  expect(url.searchParams.getAll('compareModes')).toEqual(['MT']);
  expect(url.searchParams.getAll('compareSuites')).toEqual(['']);
  expect(url.searchParams.getAll('targets')).toEqual([]);
  expect(url.searchParams.getAll('modes')).toEqual(['']);
  probe.update((state) => state.openComparison({ candidateId: 'attempt-2', baselineId: 'attempt-1', target: 'gfx1250', mode: 'ST', suites: ['memory'] }));
  expect(probe.state.filters.suites).toEqual(['memory']);
});

it('restores the entire route on Back/Forward without pushing or losing local page state', () => {
  const probe = mountState({ saved: { targets: ['gfx950'], suites: [], modes: ['ST'] } });
  const branch = { branch: 'feat/back', candidateId: 'absent-candidate', referenceId: 'absent-reference', manual: true, target: 'gfx950', mode: 'MT', detail: true };
  probe.update((state) => state.setTab('branch'));
  probe.update((state) => state.setBranchSelection(branch));
  probe.update((state) => state.openComparison({ candidateId: 'compare-c', baselineId: 'compare-b', target: 'gfx1250', mode: 'MT' }));
  probe.update((state) => {
    state.setTargets([]);
    state.setSuites([]);
    state.setHistoryRange('1M');
    state.setExplorerRunIds(['unpublished-id']);
  });
  expect(probe.browser.history.pushState).toHaveBeenCalledTimes(3);
  const pushes = probe.browser.history.pushState.mock.calls.length;
  const replacements = probe.browser.history.replaceState.mock.calls.length;
  probe.go(-1);
  expect(probe.state).toMatchObject({ tab: 'branch', branchSelection: branch, comparisonCandidateId: null, comparisonBaselineId: null, historyRange: 'ALL' });
  expect(probe.state.filters).toEqual({ targets: ['gfx950'], suites: [], modes: ['ST'] });
  probe.go(-1);
  expect(probe.state.branchSelection.candidateId).toBeNull();
  probe.go(-1);
  expect(probe.state.tab).toBe('overview');
  expect(probe.state.filters).toEqual({ targets: ['gfx950'], suites: [], modes: ['ST'] });
  probe.go(3);
  expect(probe.state).toMatchObject({ tab: 'compare', branchSelection: branch, comparisonCandidateId: 'compare-c', comparisonBaselineId: 'compare-b', historyRange: '1M', explorerRunIds: ['unpublished-id'] });
  expect(probe.state.filters).toEqual({ targets: [], suites: [], modes: ['MT'] });
  expect(probe.browser.history.pushState).toHaveBeenCalledTimes(pushes);
  expect(probe.browser.history.replaceState).toHaveBeenCalledTimes(replacements);
  expect(probe.browser.history.state).toEqual({ external: 'retained' });
});

it('replaces filter/range entries and resolves consecutive mode updaters against the latest selection', () => {
  const probe = mountState();
  probe.browser.history.replaceState.mockClear();
  probe.update((state) => {
    state.setModes([]);
    state.setModes((current) => [...current, 'ST']);
    state.setModes((current) => [...current, 'MT']);
    state.setTargets([]);
    state.setSuites([]);
    state.setHistoryRange((current) => current === 'ALL' ? '1W' : 'ALL');
  });
  expect(probe.state.filters).toEqual({ targets: [], suites: [], modes: ['ST', 'MT'] });
  expect(probe.state.historyRange).toBe('1W');
  expect(probe.browser.history.pushState).not.toHaveBeenCalled();
  expect(probe.browser.history.replaceState).toHaveBeenCalledTimes(6);
  expect(probe.entries).toHaveLength(1);
  expect(JSON.parse(probe.browser.localStorage.getItem('rocjitsu-dashboard-filters'))).toEqual(probe.state.filters);
  const url = new URL(probe.browser.location.href);
  expect(url.searchParams.getAll('targets')).toEqual(['']);
  expect(url.searchParams.getAll('suites')).toEqual(['']);
  expect(url.searchParams.getAll('modes')).toEqual(['ST', 'MT']);
});

it('changes a comparison pair atomically without navigating away from the current tab', () => {
  const probe = mountState({ href: 'https://example.test/?view=benchmarks&compareCandidate=missing-c&compareBaseline=missing-b' });
  expect(probe.state.comparisonCandidateId).toBe('missing-c');
  expect(probe.state.comparisonBaselineId).toBe('missing-b');
  probe.update((state) => state.setComparisonPair({ candidateId: 'next-c', baselineId: 'next-b' }));
  expect(probe.state).toMatchObject({ tab: 'benchmarks', comparisonCandidateId: 'next-c', comparisonBaselineId: 'next-b' });
  expect(probe.browser.history.pushState).toHaveBeenCalledTimes(1);
  probe.update((state) => state.setComparisonPair((current) => ({ candidateId: current.baselineId, baselineId: current.candidateId })));
  expect(probe.state).toMatchObject({ comparisonCandidateId: 'next-b', comparisonBaselineId: 'next-c' });
  expect(probe.browser.history.pushState).toHaveBeenCalledTimes(2);
  probe.update((state) => {
    state.setComparisonCandidateId((current) => `${current}-changed`);
    state.setComparisonBaselineId(null);
  });
  expect(probe.state).toMatchObject({ comparisonCandidateId: 'next-b-changed', comparisonBaselineId: null });
});

it('clears a malformed route error without resetting its empty scope or chosen missing identity', () => {
  const probe = mountState({ href: 'https://example.test/app/?campaign=keep&view=branch&run=missing&modes=invalid&suites=#anchor', saved: { modes: ['ST'], suites: ['gemm'] } });
  expect(probe.state.routeError).toContain('modes');
  expect(probe.state.filters).toEqual({ targets: data.targets, suites: [], modes: [] });
  expect(probe.browser.history.replaceState).not.toHaveBeenCalled();
  probe.update((state) => state.clearRouteError());
  expect(probe.state.routeError).toBe('');
  expect(probe.state.branchSelection.candidateId).toBe('missing');
  expect(probe.state.filters.modes).toEqual([]);
  expect(probe.browser.history.pushState).not.toHaveBeenCalled();
  expect(probe.browser.history.replaceState).toHaveBeenCalledTimes(1);
  const url = new URL(probe.browser.location.href);
  expect(url.searchParams.getAll('modes')).toEqual(['']);
  expect(url.searchParams.get('campaign')).toBe('keep');
  expect(url.hash).toBe('#anchor');
});

it('waits for declared data options without locking bootstrap emptiness in as preferences', () => {
  const bootstrap = { targets: [], suites: [] };
  const probe = mountState({ dataset: bootstrap, href: 'https://example.test/?view=branch&run=missing&reference=gone' });
  expect(probe.state.filters).toEqual({ targets: [], suites: [], modes: [] });
  expect(JSON.parse(probe.browser.localStorage.getItem('rocjitsu-dashboard-filters'))).toEqual({ targets: null, suites: null, modes: null });
  probe.rerender(data);
  expect(probe.state.filters).toEqual(data);
  expect(probe.state.targets.flatMap((target) => probe.state.modes.map((mode) => `${target}:${mode}`))).toEqual(['gfx950:ST', 'gfx950:MT', 'gfx1250:ST', 'gfx1250:MT']);
  probe.update((state) => { state.setTargets([]); state.setSuites([]); state.setModes([]); });
  probe.rerender(bootstrap);
  probe.rerender(data);
  expect(probe.state.filters).toEqual({ targets: [], suites: [], modes: [] });
  expect(probe.state.branchSelection).toMatchObject({ candidateId: 'missing', referenceId: 'gone' });
  expect(probe.browser.history.pushState).not.toHaveBeenCalled();
});

it('defaults to all and only declared available options', () => {
  const available = { targets: ['gfx950'], suites: ['memory'], modes: ['MT'] };
  expect(readState(available).filters).toEqual({ targets: ['gfx950'], suites: ['memory'], modes: ['MT'] });
});

it('keeps navigation and filters usable when the storage getter is denied', () => {
  const probe = mountState({ blockedStorage: true });
  expect(probe.state.filters).toEqual({ targets: data.targets, suites: data.suites, modes: data.modes });
  probe.update((state) => { state.setModes([]); state.setTab('branch'); });
  expect(probe.state.modes).toEqual([]);
  expect(probe.state.tab).toBe('branch');
  expect(probe.browser.history.pushState).toHaveBeenCalledTimes(1);
});

it('preserves exact URL scopes and missing comparison IDs across empty reload data', () => {
  const probe = mountState({ href: 'https://example.test/?view=compare&targets=gfx950&suites=memory&modes=MT&compareCandidate=unpublished-c&compareBaseline=unpublished-b', saved: { targets: ['gfx1250'], suites: ['gemm'], modes: ['ST'] } });
  const expected = { targets: ['gfx950'], suites: ['memory'], modes: ['MT'] };
  expect(probe.state.filters).toEqual(expected);
  probe.rerender({ targets: [], suites: [], modes: [] });
  expect(probe.state.filters).toEqual({ targets: [], suites: [], modes: [] });
  probe.rerender(data);
  expect(probe.state.filters).toEqual(expected);
  expect(probe.state).toMatchObject({ tab: 'compare', comparisonCandidateId: 'unpublished-c', comparisonBaselineId: 'unpublished-b' });
  expect(JSON.parse(probe.browser.localStorage.getItem('rocjitsu-dashboard-filters'))).toEqual(expected);
  expect(probe.browser.history.pushState).not.toHaveBeenCalled();
});

it('keeps user choices when a storage write fails', () => {
  const probe = mountState();
  probe.browser.localStorage.setItem.mockImplementation(() => { throw new Error('Storage quota exceeded'); });
  probe.update((state) => state.setModes([]));
  expect(probe.state.modes).toEqual([]);
  expect(new URL(probe.browser.location.href).searchParams.getAll('modes')).toEqual(['']);
});

it('takes explicit URL modes over stored preferences and exposes an atomic branch selection', () => {
  vi.stubGlobal('window', { location: { href: 'https://example.test/?view=branch&modes=&branch=feat%2Fa&run=exact' }, localStorage: { getItem: () => JSON.stringify({ modes: ['ST'] }) } });
  const state = readState();
  expect(state.filters.modes).toEqual([]);
  expect(state.tab).toBe('branch');
  expect(state.branchSelection).toMatchObject({ branch: 'feat/a', candidateId: 'exact' });
});
it('restores explicitly empty filters rather than defaults', () => {
  vi.stubGlobal('window', { localStorage: { getItem: () => JSON.stringify({ targets: [], suites: [] }) } });
  expect(readState().filters).toEqual({ targets: [], suites: [], modes: data.modes });
});
it('defaults only absent or malformed preferences from available canonical data', () => {
  vi.stubGlobal('window', { localStorage: { getItem: () => '{broken' } });
  expect(readState().filters).toEqual({ targets: data.targets, suites: data.suites, modes: data.modes });
});
it('ignores stale choices without replacing a deliberate empty scope', () => {
  vi.stubGlobal('window', { localStorage: { getItem: () => JSON.stringify({ targets: ['old'], suites: ['memory'] }) } });
  expect(readState().filters).toEqual({ targets: [], suites: ['memory'], modes: data.modes });
});
