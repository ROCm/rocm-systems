import { expect, test } from 'vitest';
import { safePublishedUrl } from '../../src/data/dashboardDataAdapter.js';
import { safeExternalUrl } from '../../src/components/branch/branchPresentation.js';
import { commitUrl } from '../../src/utils/commitUrl.js';

const sha = 'a'.repeat(40);
test.each([
  'https://example.com/repo', 'http://example.com/repo', 'HTTPS://example.com/repo',
])('link wrappers accept HTTP(S) without credentials: %s', (url) => {
  expect(safePublishedUrl(url)).toBe(true);
  expect(safeExternalUrl(url)).toBe(url);
  expect(commitUrl(url, sha)).toBe(`http${url.toLowerCase().startsWith('https') ? 's' : ''}://example.com/repo/commit/${sha}`);
});

test.each([
  null, undefined, 42, {}, '', 'not a URL', '//example.com/repo', '/repo',
  'ftp://example.com/repo', 'javascript:alert(1)',
  'https://user@example.com/repo', 'http://user:password@example.com/repo',
])('link wrappers reject unsafe URL %j', (url) => {
  expect(safePublishedUrl(url)).toBe(false);
  expect(safeExternalUrl(url)).toBeNull();
  expect(commitUrl(url, sha)).toBeNull();
});

test('wrappers retain their different prefix and construction responsibilities', () => {
  expect(safePublishedUrl(' https://example.com/repo ')).toBe(true);
  expect(safeExternalUrl(' https://example.com/repo ')).toBeNull();
  expect(commitUrl('https://example.com/repo/?q=1#part', sha)).toBe(`https://example.com/repo/commit/${sha}`);
  expect(commitUrl('https://example.com/repo', 'not-a-sha')).toBeNull();
});
