// Resolve the base URL the built application fetches published dashboard data from.

// The data-only branch keeps benchmark data under the dashboard directory.
export const PUBLISHED_DATA_PATH = 'gh-pages-rocjitsu/rocjitsu-dashboard/data';
export const PAGES_MODE = 'pages';
export const LOCAL_DATA_BASE_URL = './data/';

export function publishedDataBaseUrl(repository) {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repository ?? '')) {
    throw new Error(`Expected an <owner>/<repository> GitHub repository: ${repository}`);
  }
  return `https://raw.githubusercontent.com/${repository}/refs/heads/${PUBLISHED_DATA_PATH}/`;
}

export function resolveDataBaseUrl(mode, env = process.env) {
  const configured = env.VITE_DASHBOARD_DATA_BASE_URL?.trim();
  if (configured) return configured;
  // Every other mode serves data next to the application: fixtures from publicDir, a local
  // directory mounted at /data/, and a plain build from the site's own data/ directory.
  if (mode !== PAGES_MODE) return LOCAL_DATA_BASE_URL;
  if (!env.GITHUB_REPOSITORY) {
    throw new Error(
      `The ${PAGES_MODE} build mode needs GITHUB_REPOSITORY or VITE_DASHBOARD_DATA_BASE_URL`,
    );
  }
  return publishedDataBaseUrl(env.GITHUB_REPOSITORY);
}
