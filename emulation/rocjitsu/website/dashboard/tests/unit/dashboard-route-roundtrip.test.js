import { describe, expect, it } from 'vitest';
import * as dashboardState from '../../src/hooks/useDashboardState.js';
import { loadDashboardData, validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

it('UI-002 round-trips exact long validated attempt and branch identities', () => {
  const source = createSchema2Publication();
  const run = source.runs.find(({ id }) => id === 'fictional-branch-01');
  run.id = `fictional-${'x'.repeat(192)}`;
  run.source.branch = `fictional/${'b'.repeat(192)}`;
  source.index.runFiles = source.runs.map(({ id }) => `runs/${id}.json`);
  expect(validatePublishedDashboardData(source).data.allRuns.some(({ runId }) => runId === run.id)).toBe(true);
  const route = dashboardState.readDashboardRoute('https://example.test/app/?view=branch');
  const selection = { ...route, branchSelection: { ...route.branchSelection, branch: run.source.branch, candidateId: run.id, referenceId: run.id, manual: true }, comparisonCandidateId: run.id, comparisonBaselineId: run.id };
  const restored = dashboardState.readDashboardRoute(dashboardState.buildDashboardUrl('https://example.test/app/', selection));
  expect(restored).toEqual(selection);
  expect(restored.routeError).toBe('');
  const reloaded = loadDashboardData(JSON.parse(JSON.stringify(validatePublishedDashboardData(source).data)));
  for (const id of [restored.branchSelection.candidateId, restored.branchSelection.referenceId, restored.comparisonCandidateId, restored.comparisonBaselineId]) {
    expect(reloaded.allRuns.find(({ runId }) => runId === id)?.runId).toBe(run.id);
  }
});

it.each(['targets', 'suites'])('round-trips validated long %s through URL serialization and restoration', (key) => {
  const source = createSchema2Publication();
  const longTarget = `gfx${'9'.repeat(240)}`;
  const longSuite = `Suite ${'workload '.repeat(30)}& details + / test`;
  for (const catalog of Object.values(source.catalogs)) {
    if (key === 'suites') {
      for (const test of catalog.tests) if (test.suite === 'Triton') test.suite = longSuite;
    } else {
      catalog.configurations = Object.fromEntries(Object.entries(catalog.configurations)
        .map(([name, ids]) => [name.replace('gfx1250:', `${longTarget}:`), ids]));
    }
  }
  if (key === 'targets') {
    for (const run of source.runs) {
      for (const configuration of run.configurations) {
        if (configuration.target === 'gfx1250') configuration.target = longTarget;
      }
    }
  }
  const { data } = validatePublishedDashboardData(source);
  const value = key === 'targets' ? longTarget : longSuite;
  expect(value.length).toBeGreaterThan(200);
  expect(data[key]).toContain(value);
  const route = dashboardState.readDashboardRoute('https://example.test/app/');
  const selection = { ...route, preferences: { ...route.preferences, [key]: data[key] } };
  const url = dashboardState.buildDashboardUrl('https://example.test/app/', selection);
  expect(new URL(url).searchParams.getAll(key)).toEqual(data[key]);
  const restored = dashboardState.readDashboardRoute(url);
  expect(restored.routeError).toBe('');
  expect(restored).toEqual(selection);
  expect(restored.preferences[key].filter((selected) => data[key].includes(selected))).toEqual(data[key]);
});

describe('static-hosted dashboard routes', () => {
  it('writes an atomic selection while preserving unrelated query parameters and hash', () => {
    expect(typeof dashboardState.buildDashboardUrl).toBe('function');
    const snapshot = dashboardState.readDashboardRoute('https://example.test/app/?view=branch&branch=feat%2Fx&run=run-2&reference=run-1&manual=1&detail=1&target=gfx950&mode=MT&modes=');
    const url = new URL(dashboardState.buildDashboardUrl('https://example.test/app/?campaign=keep&view=overview#anchor', snapshot));
    expect(url.searchParams.get('campaign')).toBe('keep');
    expect(url.hash).toBe('#anchor');
    expect(dashboardState.readDashboardRoute(url.href)).toEqual(snapshot);
  });
  it('reads an exact branch attempt pair and explicit empty mode selection', () => {
    expect(typeof dashboardState.readDashboardRoute).toBe('function');
    const route = dashboardState.readDashboardRoute('https://example.test/app/?view=branch&branch=feat%2Fscheduler&run=attempt-2&reference=attempt-1&manual=1&target=gfx950&mode=MT&detail=1&modes=');
    expect(route.tab).toBe('branch');
    expect(route.branchSelection).toMatchObject({ branch: 'feat/scheduler', candidateId: 'attempt-2', referenceId: 'attempt-1', manual: true, target: 'gfx950', mode: 'MT', detail: true });
    expect(route.preferences.modes).toEqual([]);
  });
  it('keeps absent filters uninitialized and defaults safely without a browser', () => {
    const route = dashboardState.readDashboardRoute();
    expect(route).toMatchObject({ tab: 'overview', historyRange: 'ALL', preferences: { targets: null, suites: null, modes: null }, comparisonCandidateId: null, comparisonBaselineId: null, routeError: '' });
    expect(route.branchSelection).toEqual({ branch: null, candidateId: null, referenceId: null, manual: false, target: 'gfx1250', mode: 'ST', detail: false });
  });
  it('round-trips repeated scopes without treating an explicit empty as absent', () => {
    const route = dashboardState.readDashboardRoute('https://example.test/app/?targets=gfx950&targets=gfx1250&suites=gemm&suites=memory&modes=ST&modes=MT&compareCandidate=unpublished-c&compareBaseline=unpublished-b');
    expect(route.preferences).toEqual({ targets: ['gfx950', 'gfx1250'], suites: ['gemm', 'memory'], modes: ['ST', 'MT'] });
    expect(dashboardState.readDashboardRoute(dashboardState.buildDashboardUrl('https://example.test/app/', route))).toEqual(route);
    const empty = dashboardState.readDashboardRoute('https://example.test/?targets=&suites=&modes=');
    expect(empty.preferences).toEqual({ targets: [], suites: [], modes: [] });
    expect(empty.routeError).toBe('');
  });
  it.each([
    ['targets=gfx950&targets=gfx950', 'targets'],
    ['suites=&suites=gemm', 'suites'],
    ['modes=st', 'modes'],
    ['modes=ST&MT=keep&modes=ST', 'modes'],
  ])('fails closed for malformed repeated scope %s', (query, key) => {
    const route = dashboardState.readDashboardRoute(`https://example.test/?${query}`);
    expect(route.preferences[key]).toEqual([]);
    expect(route.routeError).toContain(key);
  });
  it.each(['view=removed', 'range=2Y', 'mode=unknown', 'manual=maybe', 'detail=2', 'run=first&run=second'])('reports malformed scalar selection %s', (query) => {
    expect(dashboardState.readDashboardRoute(`https://example.test/?${query}`).routeError).not.toBe('');
  });
  it('reports an invalid URL rather than crashing dashboard initialization', () => {
    const route = dashboardState.readDashboardRoute('not an absolute URL');
    expect(route.tab).toBe('overview');
    expect(route.routeError).toContain('URL');
  });
});
