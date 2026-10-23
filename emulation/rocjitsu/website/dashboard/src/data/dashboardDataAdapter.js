import { isSafeHttpUrl } from '../utils/externalUrls.js';
import { isSupportedNumber, isTimingNumber, typedJsonIdentity } from './jsonValues.js';
import { CURRENT_RUN_SCHEMA_VERSION, runSchemaVersion, THREAD_POLICIES } from './runSchema.js';
import { DASHBOARD_SITE_CONFIG } from '../config/siteConfig.js';
import { backfillRunIds, compareRunExecution, sortRunsByCommit } from './runOrdering.js';

export const RUN_FILE_PATTERN = /^runs\/(?:default-branch|side-branches)\/[A-Za-z0-9._-]+\.json$/;
const text = (value) => typeof value === 'string' && Boolean(value.trim());
const token = (value) => text(value) && /^[A-Za-z0-9._-]+$/.test(value) && !['.', '..'].includes(value);
const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

export function safePublishedUrl(value) {
  if (!text(value)) return false;
  try {
    const url = new URL(value);
    return isSafeHttpUrl(url);
  } catch { return false; }
}

export function validatePublishedManifest(index) {
  if (!Array.isArray(index?.runFiles)) throw new Error('Expected index runFiles array');
  for (const name of index.runFiles) {
    if (typeof name !== 'string' || !RUN_FILE_PATTERN.test(name) || ['.', '..'].includes(name.split('/').at(-1).slice(0, -5))) throw new Error(`Invalid run filename: ${String(name)}`);
  }
  if (new Set(index.runFiles).size !== index.runFiles.length) throw new Error('Duplicate run filename');
}

export function validateDashboardSiteConfig(siteConfig) {
  if (!safePublishedUrl(siteConfig?.repository)) throw new Error('Expected dashboard site configuration to contain a safe HTTP repository URL');
  if (typeof siteConfig.isBeta !== 'boolean' || siteConfig.canonicalBranch !== 'develop') throw new Error('Expected site configuration isBeta boolean and canonicalBranch develop');
}

function normalizedEnvironment(provenance, summary) {
  const details = new Map();
  for (const [section, values] of Object.entries(provenance)) {
    if (!object(values)) continue;
    for (const [key, value] of Object.entries(values)) {
      if (section === 'rocjitsu' && ['rocjitsu_commit_sha', 'rocjitsu_commit_timestamp', 'target_config_sha256'].includes(key)) continue;
      const name = `${section}.${key}`;
      details.set(name, { key: name, label: name, value: value === null || ['string', 'boolean'].includes(typeof value) || isSupportedNumber(value) ? value : typedJsonIdentity(value) });
    }
  }
  for (const key of ['warmups', 'samples', 'timeout_seconds', 'benchmark_suite']) {
    const name = `execution.${key}`;
    details.set(name, { key: name, label: name, value: summary[key] });
  }
  return [...details.values()].sort((a, b) => a.key.localeCompare(b.key));
}

function normalizeResult(row) {
  if (!object(row) || !text(row.id) || !token(row.target) || !THREAD_POLICIES.includes(row.thread_policy)) throw new Error('Unusable benchmark result identity');
  if (!text(row.name)) throw new Error('Unusable benchmark result name');
  let durationSeconds = null;
  let unavailableReason = null;
  if (row.status === 'completed') {
    if (Array.isArray(row.timing_results_s) && row.timing_results_s.length && row.timing_results_s.every(isTimingNumber)) {
      const samples = [...row.timing_results_s].sort((a, b) => a - b);
      const middle = Math.floor(samples.length / 2);
      durationSeconds = samples.length % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) / 2;
    } else unavailableReason = 'No usable completed timing samples';
  } else unavailableReason = row.error ?? `Result ${row.status ?? 'unavailable'}`;
  return { ...row, testId: `${row.target}/${row.thread_policy}:${row.id}`, logicalTestId: row.id, mode: row.thread_policy, durationSeconds, unavailableReason };
}

function normalizeRun(run) {
  const publication = run.publication;
  const source = publication.source;
  const summary = run.execution_summary;
  const provenance = run.provenance ?? {};
  const native = provenance.rocjitsu ?? {};
  const environment = normalizedEnvironment(provenance, summary);
  const pr = source.pull_request;
  return {
    runId: publication.id, catalogId: summary.benchmark_suite,
    timestamp: summary.finished_at, commitTimestamp: native.rocjitsu_commit_timestamp,
    trigger: publication.trigger, machineId: publication.machine_label, branch: source.branch,
    ...(object(source.base) ? { sourceBase: { ...source.base } } : {}),
    ...(object(pr) ? { pullRequest: { ...pr, ...(pr.url && !safePublishedUrl(pr.url) ? { url: undefined } : {}) } } : {}),
    environmentId: JSON.stringify(environment.map(({ key, value }) => [key, value])),
    provenance: { rocjitsuCommitSha: native.rocjitsu_commit_sha, commitMessage: source.message, details: [...environment,
      { key: 'rocjitsu.workflow_commit_sha', label: 'Workflow commit', value: publication.workflow_commit_sha },
      { key: 'auxiliary.tensile_commit_sha', label: 'Tensile commit', value: publication.tensile_commit_sha },
    ] },
    targets: [...new Set(run.benchmark_results.map(({ target }) => target))],
    modes: THREAD_POLICIES.filter((policy) => run.benchmark_results.some((row) => row.thread_policy === policy)),
    configurations: [...new Map(run.benchmark_results.map(({ target, thread_policy }) => [`${target}/${thread_policy}`, { target, thread_policy }])).values()],
    tests: run.benchmark_results.map(normalizeResult),
  };
}

export function adaptPublishedDashboardData({ index, runs, runErrors = [], siteConfig = DASHBOARD_SITE_CONFIG }) {
  validatePublishedManifest(index);
  validateDashboardSiteConfig(siteConfig);
  if (!Array.isArray(runs) || runs.length !== index.runFiles.length) throw new Error('Loaded runs must match index runFiles');
  const failures = runErrors.flatMap((error, position) => error ? [`- ${index.runFiles[position]}: ${error.message}`] : []);
  if (failures.length) throw new Error(`Unable to load dashboard runs:\n${failures.join('\n')}`);
  const ids = new Set();
  const catalogs = new Map();
  const definitions = new Map();
  const normalized = runs.map((run, position) => {
    if (runErrors[position]) throw runErrors[position];
    runSchemaVersion(run);
    const publication = run.publication;
    if (!token(publication?.id) || !text(publication?.source?.branch)) throw new Error('Run requires publication ID and branch');
    if (ids.has(publication.id)) throw new Error(`Duplicate run ID ${publication.id}`);
    ids.add(publication.id);
    const directory = publication.source.branch === siteConfig.canonicalBranch ? 'default-branch' : 'side-branches';
    if (index.runFiles[position] !== `runs/${directory}/${publication.id}.json`) throw new Error(`Run ${publication.id} filename must match its ID and branch directory`);
    if (!text(run.execution_summary?.benchmark_suite) || !Array.isArray(run.benchmark_results)) throw new Error('Run requires benchmark suite and benchmark_results array');
    const suite = run.execution_summary.benchmark_suite;
    if (!catalogs.has(suite)) {
      const tests = new Map();
      const configurations = {};
      for (const row of run.benchmark_results) {
        if (!tests.has(row.id)) tests.set(row.id, { id: row.id, suite: row.suite, name: row.name, operation: row.operation, problem: row.problem });
        const key = `${row.target}/${row.thread_policy}`;
        if (!Object.hasOwn(configurations, key)) Object.defineProperty(configurations, key, { value: [], enumerable: true });
        configurations[key].push(row.id);
      }
      const catalog = { id: suite, tests: [...tests.values()], configurations };
      catalogs.set(suite, catalog);
      for (const definition of catalog.tests) definitions.set(definition.id, definition);
    }
    return normalizeRun(run);
  });
  const allRuns = normalized.sort(compareRunExecution);
  const canonical = allRuns.filter(({ branch }) => branch === siteConfig.canonicalBranch);
  return { sourceData: { index, runs }, data: {
    ...siteConfig, schemaVersion: CURRENT_RUN_SCHEMA_VERSION, generatedAt: index.generatedAt,
    allRuns, runs: canonical, targets: [...new Set(allRuns.flatMap(({ targets }) => targets))].sort(), suites: [...new Set([...definitions.values()].map(({ suite }) => suite))].sort(), modes: [...THREAD_POLICIES], testCatalog: [...definitions.values()], catalogs: Object.fromEntries(catalogs),
    latestRun: canonical.at(-1) ?? null, latestCommitRun: sortRunsByCommit(canonical).at(-1) ?? null, backfillRunIds: backfillRunIds(canonical),
  } };
}
