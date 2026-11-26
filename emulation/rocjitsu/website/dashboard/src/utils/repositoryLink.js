import { repositoryBase, repositoryPath } from './externalUrls.js';

// The dashboard lives in this subdirectory of the published repository.
const PROJECT_PATH = 'emulation/rocjitsu';

// A branch name may contain slashes, but no empty, `.` or `..` segment, which would climb out of /tree/.
const safeBranchPath = (branch) => {
  if (typeof branch !== 'string' || !branch) return null;
  const segments = branch.split('/');
  return segments.some((segment) => !segment || segment === '.' || segment === '..')
    ? null : segments.map(encodeURIComponent).join('/');
};

// Header link to the rocjitsu source tree, derived from the publication's configured repository and
// canonical branch so another publication points at its own repository. Null when unusable.
export function repositoryLink(repository, canonicalBranch = 'develop') {
  const url = repositoryBase(repository);
  const branchPath = safeBranchPath(canonicalBranch);
  if (!url || !branchPath) return null;
  const label = url.pathname.split('/').filter(Boolean).join('/') || url.host;
  url.pathname = repositoryPath(url, `/tree/${branchPath}/${PROJECT_PATH}`);
  return { href: url.href, label };
}
