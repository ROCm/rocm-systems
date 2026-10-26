import { isSafeHttpUrl } from './externalUrls.js';

export function commitUrl(repository, sha) {
  if (typeof sha !== 'string' || !/^[a-fA-F0-9]{7,40}$/.test(sha) || typeof repository !== 'string') return null;
  try {
    const url = new URL(repository);
    if (!isSafeHttpUrl(url)) return null;
    url.pathname = `${url.pathname.replace(/\/$/, '')}/commit/${sha}`;
    url.search = '';
    url.hash = '';
    return url.href;
  } catch {
    return null;
  }
}
