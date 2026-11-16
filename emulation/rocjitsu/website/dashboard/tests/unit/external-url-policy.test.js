import { expect, test } from 'vitest';
import { safePublishedUrl } from '../../src/data/dashboardDataAdapter.js';
import { safeExternalUrl } from '../../src/components/branch/branchPresentation.js';
import { commitUrl } from '../../src/utils/commitUrl.js';
import { repositoryLink } from '../../src/utils/repositoryLink.js';

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

test.each([null, '', '../../x', 'a//b', 'a/./b', 'a/../b', 42])('repository link rejects unusable branch %j', (branch) => {
  expect(repositoryLink('https://github.com/ROCm/rocm-systems', branch)).toBeNull();
});

test.each(['https://example.test/', 'https://example.test'])('an origin-only repository %s gives single-slash commit and tree links', (repository) => {
  expect(commitUrl(repository, 'abcdef0')).toBe('https://example.test/commit/abcdef0');
  expect(repositoryLink(repository, 'develop')).toEqual({ href: 'https://example.test/tree/develop/emulation/rocjitsu', label: 'example.test' });
});

test('a repository with a path keeps it, with and without trailing slashes', () => {
  expect(commitUrl('https://example.test/org/repo/', 'abcdef0')).toBe('https://example.test/org/repo/commit/abcdef0');
  expect(repositoryLink('https://example.test/org/repo//', 'main').href).toBe('https://example.test/org/repo/tree/main/emulation/rocjitsu');
});

test('repository link encodes each branch segment and keeps slashes between them', () => {
  expect(repositoryLink('https://github.com/ROCm/rocm-systems', 'users/a b/c?d').href)
    .toBe('https://github.com/ROCm/rocm-systems/tree/users/a%20b/c%3Fd/emulation/rocjitsu');
});

test.each(['https://github.com/ROCm/rocm-systems.git', 'https://github.com/ROCm/rocm-systems.git/', 'https://github.com/ROCm/rocm-systems//'])('commit and repository links agree on the repository base of %s', (repository) => {
  expect(repositoryLink(repository).label).toBe('ROCm/rocm-systems');
  expect(repositoryLink(repository).href).toBe('https://github.com/ROCm/rocm-systems/tree/develop/emulation/rocjitsu');
  expect(commitUrl(repository, sha)).toBe(`https://github.com/ROCm/rocm-systems/commit/${sha}`);
});
