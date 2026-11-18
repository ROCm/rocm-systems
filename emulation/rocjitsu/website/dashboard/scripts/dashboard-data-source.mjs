// Resolve the base URL the built application fetches published dashboard data from.

// The data-only branch keeps benchmark data under the dashboard directory.
export const PUBLISHED_DATA_BASE_URL = (
  'https://raw.githubusercontent.com/ROCm/rocm-systems/'
  + 'refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/'
);
export const LOCAL_DATA_BASE_URL = './data/';

export function resolveDataBaseUrl(mode, env = process.env) {
  if (mode === 'production' && !env.DASHBOARD_DATA_DIR?.trim()) {
    return PUBLISHED_DATA_BASE_URL;
  }
  return LOCAL_DATA_BASE_URL;
}
