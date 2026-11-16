import { expect, test } from 'vitest';
import { readDashboardRoute, buildDashboardUrl } from '../../src/hooks/useDashboardState.js';

test('literal thread policies roundtrip while retaining unrelated URL state', () => {
  const href = 'https://example.test/dashboard?modes=single&modes=default&mode=default&external=keep#section';
  const route = readDashboardRoute(href);
  expect(route.routeError).toBe('');
  expect(route.preferences.modes).toEqual(['single', 'default']);
  expect(route.branchSelection.mode).toBe('default');
  const url = new URL(buildDashboardUrl(href, route));
  expect(url.searchParams.getAll('modes')).toEqual(['single', 'default']);
  expect(url.searchParams.get('external')).toBe('keep');
  expect(url.hash).toBe('#section');
});

test.each(['ST', 'MT'])('retired policy %s is rejected rather than aliased', (mode) => {
  const route = readDashboardRoute(`https://example.test/?modes=${mode}&mode=${mode}`);
  expect(route.routeError).toContain('Invalid');
  expect(route.preferences.modes).toEqual([]);
  expect(route.branchSelection.mode).toBeNull();
});
