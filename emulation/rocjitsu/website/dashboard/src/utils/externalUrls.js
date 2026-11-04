// Shared policy for an already-parsed external link. Callers own parsing,
// stricter input rules, and any URL construction.
export function isSafeHttpUrl(url) {
  return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password;
}

// The published repository as a safe URL whose path has no trailing slashes or `.git`, or null.
// Callers append their own path (`/commit/<sha>`, `/tree/<branch>`).
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
