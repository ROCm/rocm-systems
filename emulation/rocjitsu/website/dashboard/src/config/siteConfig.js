// Website-owned metadata
import metadata from './metadata.json' with { type: 'json' };

export const DASHBOARD_SITE_CONFIG = Object.freeze({
  repository: metadata.repository,
  isBeta: metadata.isBeta,
  canonicalBranch: metadata.canonicalBranch,
});
