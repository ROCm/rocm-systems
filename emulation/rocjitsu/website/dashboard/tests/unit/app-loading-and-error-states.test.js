import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, expect, it, vi } from 'vitest';
import App from '../../src/App.jsx';

vi.mock('../../src/data/publishedDataUrls.js', () => ({ resolvePublishedDataUrls: () => ({ indexUrl: 'https://example.test/data/index.json' }) }));
vi.mock('../../src/components/views/BranchRunsView.jsx', () => ({ default: () => 'branch-surface-without-published-data' }));
afterEach(() => vi.unstubAllGlobals());
it('renders a usable shell when browser storage is denied', () => {
  vi.stubGlobal('window', { location: { href: 'https://example.test/' }, get localStorage() { throw new Error('Storage denied'); }, matchMedia: () => ({ matches: false }) });
  expect(() => renderToStaticMarkup(createElement(App))).not.toThrow();
});
it('discloses malformed URL filters even while data is loading', () => {
  vi.stubGlobal('window', { location: { href: 'https://example.test/?modes=unknown' }, localStorage: { getItem: () => null }, matchMedia: () => ({ matches: false }) });
  const html = renderToStaticMarkup(createElement(App));
  expect(html.includes('Invalid modes selection in this URL.')).toBe(true);
});
it('keeps branch navigation available but gates its contents on the validated snapshot', () => {
  vi.stubGlobal('window', {
    location: { href: 'https://example.test/?view=branch' },
    localStorage: { getItem: () => null },
    matchMedia: () => ({ matches: false }),
  });
  const html = renderToStaticMarkup(createElement(App));
  expect(html).toContain('dashboard-data-loading');
  expect(html).not.toContain('branch-surface-without-published-data');
});
it('shows copyright and license without unrelated framework attribution while loading', () => {
  vi.stubGlobal('window', { location: { href: 'https://example.test/' }, localStorage: { getItem: () => null }, matchMedia: () => ({ matches: false }) });
  const html = renderToStaticMarkup(createElement(App));
  expect(html.includes('Copyright © 2025–2026 Advanced Micro Devices, Inc.')).toBe(true);
  expect(html.includes('MIT License')).toBe(true);
  expect(html).not.toContain('Made with Material for MkDocs');
});
