// Fictional-only construction helpers. Never imported by the dashboard.
export const fixtureSha = (index) => index.toString(16).padStart(40, '0');

export function fixtureCatalog(benchmark_suite, definitions, lanes) {
  const membership = {};
  for (const [key, members] of Object.entries(lanes)) {
    const [target, policy] = key.split('/');
    membership[target] ??= {};
    membership[target][policy] = members;
  }
  return { benchmark_suite, benchmark_definitions: Object.fromEntries(definitions.map(({ id, ...definition }) => [id, definition])), membership };
}

export function fixtureProvenance(commit, committedAt, sdk = 'test-only') {
  return {
    auxiliary: { sdk, enabled: true, cores: 32 },
    corpus: { corpus_commit_sha: fixtureSha(900), corpus_commit_timestamp: '2025-01-01T00:00:00.000Z' },
    machine: { hostname: 'fictional-host', cpu: 'fictional-cpu', platform: 'fictional-platform', kernel: 'fictional-kernel' },
    rocjitsu: { rocjitsu_commit_sha: commit, rocjitsu_commit_timestamp: committedAt, rocm_sdk_version: 'fictional-sdk' },
  };
}

export function fixtureSummary(benchmark_suite, timestamp, finished_at = timestamp) {
  return { benchmark_suite, timestamp, finished_at, wall_time_s: 1, samples: 1, warmups: 0, timeout_seconds: 1200 };
}
