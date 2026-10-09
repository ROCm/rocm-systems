import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { historyRanges } from '../config/historyRanges.js';

const FILTER_STORAGE_KEY = 'rocjitsu-dashboard-filters';
const BENCHMARK_STORAGE_KEY = 'rocjitsu-dashboard-benchmarks';
const pages = new Set(['overview', 'branch', 'benchmarks', 'compare']);
const ranges = new Set(historyRanges.map(({ id }) => id));
const emptyBranchSelection = { branch: null, candidateId: null, referenceId: null, manual: false, target: null, mode: null, detail: false };

export function readDashboardRoute(href = globalThis.window?.location?.href ?? 'http://localhost/') {
  const preferences = {};
  const errors = [];
  let params;
  try { params = new URL(href).searchParams; } catch {
    params = new URLSearchParams();
    errors.push('Invalid dashboard URL.');
  }
  for (const key of ['targets', 'suites', 'modes']) {
    const values = params.getAll(key);
    if (!values.length) preferences[key] = null;
    else if (values.length === 1 && values[0] === '') preferences[key] = [];
    else if (values.some((value) => !value || (key === 'modes' && !['ST', 'MT'].includes(value))) || new Set(values).size !== values.length) {
      preferences[key] = [];
      errors.push(`Invalid ${key} selection in this URL.`);
    } else preferences[key] = values;
  }
  const field = (key, fallback, allowed) => {
    const values = params.getAll(key);
    if (!values.length) return fallback;
    const value = values[0];
    // Published attempt/branch identities have no scalar length bound. Keep
    // exact missing identities too; only duplicate/empty/enum values are invalid.
    if (values.length !== 1 || !value || (allowed && !allowed.has(value))) {
      errors.push(`Invalid ${key} selection in this URL.`);
      return fallback;
    }
    return value;
  };
  const route = {
    tab: field('view', 'overview', pages),
    historyRange: field('range', 'ALL', ranges),
    preferences,
    branchSelection: {
      ...emptyBranchSelection,
      branch: field('branch', null),
      candidateId: field('run', null),
      referenceId: field('reference', null),
      manual: field('manual', '0', new Set(['0', '1'])) === '1',
      target: field('target', null),
      mode: field('mode', null, new Set(['ST', 'MT'])),
      detail: field('detail', '0', new Set(['0', '1'])) === '1',
    },
    comparisonCandidateId: field('compareCandidate', null),
    comparisonBaselineId: field('compareBaseline', null),
  };
  return { ...route, routeError: errors.join(' ') };
}

export function buildDashboardUrl(href, snapshot) {
  const url = new URL(href);
  const params = url.searchParams;
  const fields = {
    view: snapshot.tab,
    range: snapshot.historyRange,
    branch: snapshot.branchSelection.branch,
    run: snapshot.branchSelection.candidateId,
    reference: snapshot.branchSelection.referenceId,
    manual: snapshot.branchSelection.manual ? '1' : null,
    target: snapshot.branchSelection.target,
    mode: snapshot.branchSelection.mode,
    detail: snapshot.branchSelection.detail ? '1' : null,
    compareCandidate: snapshot.comparisonCandidateId,
    compareBaseline: snapshot.comparisonBaselineId,
  };
  for (const [key, value] of Object.entries(fields)) {
    if (value == null) params.delete(key);
    else params.set(key, value);
  }
  for (const key of ['targets', 'suites', 'modes']) {
    params.delete(key);
    const values = snapshot.preferences[key];
    if (values !== null) {
      if (!values.length) params.set(key, '');
      else values.forEach((value) => params.append(key, value));
    }
  }
  return url.href;
}

function readFilters() {
  const preferences = { targets: null, suites: null, modes: null };
  try {
    const saved = JSON.parse(globalThis.window?.localStorage?.getItem(FILTER_STORAGE_KEY) ?? 'null');
    for (const key of Object.keys(preferences)) {
      if (Array.isArray(saved?.[key]) && saved[key].every((item) => typeof item === 'string' && item.length > 0 && (key !== 'modes' || ['ST', 'MT'].includes(item))) && new Set(saved[key]).size === saved[key].length) {
        preferences[key] = saved[key];
      }
    }
  } catch { /* Storage is optional, including when its getter throws. */ }
  return preferences;
}

function readBenchmarkSelection() {
  try {
    const saved = JSON.parse(globalThis.window?.localStorage?.getItem(BENCHMARK_STORAGE_KEY) ?? 'null');
    if (Array.isArray(saved) && saved.length <= 8
      && saved.every((test) => test && ['id', 'name', 'suite'].every((key) => typeof test[key] === 'string' && test[key].trim())
        && (test.problem === undefined || test.problem !== null && typeof test.problem === 'object' && !Array.isArray(test.problem)
          && Object.values(test.problem).every((value) => typeof value === 'string' || typeof value === 'boolean' || typeof value === 'number' && Number.isFinite(value))))
      && new Set(saved.map((test) => test.id)).size === saved.length) return saved;
  } catch { /* Storage is optional; missing/malformed preferences are unchosen. */ }
  return null;
}

function readInitialState() {
  const route = readDashboardRoute(globalThis.window?.location?.href ?? 'http://localhost/');
  const saved = readFilters();
  return {
    ...route,
    preferences: Object.fromEntries(Object.keys(saved).map((key) => [key, route.preferences[key] ?? saved[key]])),
  };
}

function selectFilters(data, preferences) {
  const modes = data.modes ?? [];
  return {
    targets: preferences.targets === null ? data.targets : preferences.targets.filter((target) => data.targets.includes(target)),
    suites: preferences.suites === null ? data.suites : preferences.suites.filter((suite) => data.suites.includes(suite)),
    modes: preferences.modes === null ? modes : preferences.modes.filter((mode) => modes.includes(mode)),
  };
}

function writeHistory(snapshot, method) {
  const browser = globalThis.window;
  if (!browser?.location?.href || !browser.history?.[method]) return;
  const href = buildDashboardUrl(browser.location.href, snapshot);
  if (href !== browser.location.href) browser.history[method](browser.history.state, '', href);
}

const resolveNext = (next, current) => typeof next === 'function' ? next(current) : next;

export function useDashboardState(data) {
  // One route snapshot makes branch/pair/scope transitions atomic. Null filters
  // stay uninitialized through bootstrap; [] always means an intentional empty.
  const [snapshot, setSnapshot] = useState(readInitialState);
  const snapshotRef = useRef(snapshot);
  const { preferences, branchSelection, historyRange, tab, comparisonBaselineId, comparisonCandidateId, routeError } = snapshot;
  const filters = useMemo(() => selectFilters(data, preferences), [data, preferences]);

  // Resolve updates against the last event, not a stale render. History writes
  // are never inside React's functional state updater (StrictMode replays it).
  const updateRoute = useCallback((change, method = 'pushState') => {
    const next = change(snapshotRef.current);
    snapshotRef.current = next;
    setSnapshot(next);
    writeHistory(next, method);
  }, []);
  const setBranchSelection = useCallback((next) => updateRoute((current) => ({
    ...current,
    branchSelection: { ...current.branchSelection, ...resolveNext(next, current.branchSelection) },
  })), [updateRoute]);

  const openComparison = useCallback(({ candidateId, baselineId, target, mode, suites }) => updateRoute((current) => ({
    ...current,
    tab: 'compare',
    comparisonCandidateId: candidateId,
    comparisonBaselineId: baselineId,
    preferences: {
      ...current.preferences,
      targets: [target],
      modes: [mode],
      suites: suites === undefined ? current.preferences.suites : suites,
    },
  })), [updateRoute]);

  const clearRouteError = useCallback(() => updateRoute((current) => ({ ...current, routeError: '' }), 'replaceState'), [updateRoute]);
  const setFilter = useCallback((key, next) => updateRoute((current) => ({
    ...current,
    preferences: { ...current.preferences, [key]: resolveNext(next, selectFilters(data, current.preferences)[key]) },
  }), 'replaceState'), [data, updateRoute]);
  const setTargets = useCallback((next) => setFilter('targets', next), [setFilter]);
  const setSuites = useCallback((next) => setFilter('suites', next), [setFilter]);
  const setModes = useCallback((next) => setFilter('modes', next), [setFilter]);
  const setHistoryRange = useCallback((next) => updateRoute((current) => ({ ...current, historyRange: resolveNext(next, current.historyRange) }), 'replaceState'), [updateRoute]);
  const setTab = useCallback((next) => updateRoute((current) => ({ ...current, tab: resolveNext(next, current.tab) })), [updateRoute]);
  const setComparisonBaselineId = useCallback((next) => updateRoute((current) => ({ ...current, comparisonBaselineId: resolveNext(next, current.comparisonBaselineId) })), [updateRoute]);
  const setComparisonCandidateId = useCallback((next) => updateRoute((current) => ({ ...current, comparisonCandidateId: resolveNext(next, current.comparisonCandidateId) })), [updateRoute]);
  const setComparisonPair = useCallback((next) => updateRoute((current) => {
    const pair = { candidateId: current.comparisonCandidateId, baselineId: current.comparisonBaselineId };
    const selected = { ...pair, ...resolveNext(next, pair) };
    return { ...current, comparisonCandidateId: selected.candidateId, comparisonBaselineId: selected.baselineId };
  }), [updateRoute]);
  useEffect(() => {
    const browser = globalThis.window;
    const restore = () => {
      const next = readDashboardRoute(browser.location.href);
      snapshotRef.current = next;
      setSnapshot(next);
    };
    browser?.addEventListener?.('popstate', restore);
    // Capture initial stored choices in this history entry, so Back never reads
    // preferences saved by a later entry. Do not erase a malformed incoming URL.
    if (!snapshotRef.current.routeError) writeHistory(snapshotRef.current, 'replaceState');
    return () => browser?.removeEventListener?.('popstate', restore);
  }, []);
  useEffect(() => {
    try { globalThis.window?.localStorage?.setItem(FILTER_STORAGE_KEY, JSON.stringify(preferences)); } catch { /* Storage is optional. */ }
  }, [preferences]);

  const [benchmarkMode, setBenchmarkMode] = useState('single');
  const [search, setSearch] = useState('');
  const [explorerRunIds, setExplorerRunIds] = useState([]);
  // Null is not yet chosen; [] is an explicitly empty grid. Keep full workload
  // definitions here so page/data unmounts cannot discard unavailable identity.
  const [selectedBenchmarks, setSelectedBenchmarks] = useState(readBenchmarkSelection);
  useEffect(() => {
    // Do not persist derived defaults (or empty bootstrap data) as a user choice.
    if (selectedBenchmarks === null) return;
    try { globalThis.window?.localStorage?.setItem(BENCHMARK_STORAGE_KEY, JSON.stringify(selectedBenchmarks)); } catch { /* In-memory choices still survive data reloads. */ }
  }, [selectedBenchmarks]);

  return {
    filters,
    ...filters,
    historyRange,
    tab,
    branchSelection,
    routeError,
    clearRouteError,
    benchmarkMode,
    search,
    comparisonBaselineId,
    comparisonCandidateId,
    explorerRunIds,
    selectedBenchmarks,
    setTargets,
    setSuites,
    setModes,
    setComparisonPair,
    setHistoryRange,
    setTab,
    setBranchSelection,
    openComparison,
    setBenchmarkMode,
    setSearch,
    setComparisonBaselineId,
    setComparisonCandidateId,
    setExplorerRunIds,
    setSelectedBenchmarks,
  };
}
