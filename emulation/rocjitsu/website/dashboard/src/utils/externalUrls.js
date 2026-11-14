// Shared policy for an already-parsed external link. Callers own parsing,
// stricter input rules, and any URL construction.
export function isSafeHttpUrl(url) {
  return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password;
}

// The published repository as a safe URL whose path has no trailing slashes or `.git`, or null.
// A URL always reads its empty path back as `/`, so callers join with `repositoryPath`, not by appending to `pathname`.
export function repositoryBase(repository) {
  if (typeof repository !== 'string') return null;
  try {
    const url = new URL(repository);
    if (!isSafeHttpUrl(url)) return null;
    url.pathname = url.pathname.replace(/\/+$/, '').replace(/\.git$/, '');
    url.search = '';
    url.hash = '';
    return url;
  } catch {
    return null;
  }
}

// `suffix` (for example `/commit/<sha>`) joined onto the repository path without doubling the slash an origin-only
// repository (`https://example.test/`) reads back with.
export function repositoryPath(url, suffix) {
  return `${url.pathname.replace(/\/+$/, '')}${suffix}`;
}
