import { repositoryBase, repositoryPath } from './externalUrls.js';

export function commitUrl(repository, sha) {
  if (typeof sha !== 'string' || !/^[a-fA-F0-9]{7,40}$/.test(sha)) return null;
  const url = repositoryBase(repository);
  if (!url) return null;
  url.pathname = repositoryPath(url, `/commit/${sha}`);
  return url.href;
}
