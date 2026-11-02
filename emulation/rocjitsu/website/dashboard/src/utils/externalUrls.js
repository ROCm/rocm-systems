// Shared policy for an already-parsed external link. Callers own parsing,
// stricter input rules, and any URL construction.
export function isSafeHttpUrl(url) {
  return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password;
}
