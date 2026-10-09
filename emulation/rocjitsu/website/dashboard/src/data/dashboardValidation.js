import { CURRENT_RUN_SCHEMA_VERSION, isExcludedPluginRun, migrateSchema1Run, publishedRunPath, runSchemaVersion } from './runSchema.js';
import { DASHBOARD_SITE_CONFIG } from '../config/siteConfig.js';
import { backfillRunIds, compareRunExecution, sortRunsByCommit } from './runOrdering.js';

export const CURRENT_SCHEMA_VERSION = CURRENT_RUN_SCHEMA_VERSION;
export const RUN_FILE_PATTERN = /^runs\/(?:(?:default-branch|side-branches)\/)?[A-Za-z0-9._-]+\.json$/;
export const CATALOG_FILE_PATTERN = /^test-catalogs\/[A-Za-z0-9._-]+\.json$/;
const SHA = /^[0-9a-f]{40}$/;
const TOKEN = /^[A-Za-z0-9._-]+$/;
const CONFIGURATION = /^([A-Za-z0-9._-]+):(ST|MT)$/;
const ISO = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|([+-])(\d{2}):(\d{2}))$/;
const hasText = (value) => typeof value === 'string' && Boolean(value.trim());
const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const scalar = (value) => typeof value === 'string' || typeof value === 'boolean' || typeof value === 'number' && Number.isFinite(value);
const sameSet = (left, right) => {
  if (left.length !== right.length) return false;
  const sortedLeft = [...left].sort();
  const sortedRight = [...right].sort();
  return sortedLeft.every((value, index) => value === sortedRight[index]);
};
const identity = (value) => JSON.stringify(value);
const environmentIdentity = (environment) => identity(environment.map(({ key, value }) => [key, value]).sort(([a], [b]) => a.localeCompare(b)));
const definitionIdentity = (test) => identity([test.suite, test.name, Object.entries(test.problem).sort(([a], [b]) => a.localeCompare(b))]);

export function isIsoTimestamp(value) {
  if (typeof value !== 'string') return false;
  const match = ISO.exec(value);
  if (!match || !Number.isFinite(Date.parse(value))) return false;
  const [, y, m, d, h, min, s, , oh = '0', om = '0'] = match;
  const date = new Date(`${y}-${m}-${d}T00:00:00.000Z`);
  return date.getUTCFullYear() === Number(y) && date.getUTCMonth() + 1 === Number(m)
    && date.getUTCDate() === Number(d) && Number(h) <= 23 && Number(min) <= 59
    && Number(s) <= 59 && Number(oh) <= 23 && Number(om) <= 59;
}

function safeUrl(value, github = false) {
  if (!hasText(value)) return false;
  try {
    const url = new URL(value);
    return !url.username && !url.password && ['http:', 'https:'].includes(url.protocol)
      && (!github || url.protocol === 'https:' && url.hostname === 'github.com');
  } catch { return false; }
}

export function validatePublishedManifest(index) {
  if (!isIsoTimestamp(index?.generatedAt) || !Array.isArray(index?.runFiles)) throw new Error('Expected index generatedAt strict ISO timestamp and runFiles array');
  for (const name of index.runFiles) {
    if (typeof name !== 'string' || !RUN_FILE_PATTERN.test(name)) throw new Error(`Invalid run filename: ${String(name)}`);
  }
  if (new Set(index.runFiles).size !== index.runFiles.length) throw new Error('Duplicate run filename');
}

export function validateDashboardSiteConfig(siteConfig) {
  if (!safeUrl(siteConfig?.repository)) throw new Error('Expected dashboard site configuration to contain a safe HTTP repository URL');
  if (typeof siteConfig.isBeta !== 'boolean' || siteConfig.canonicalBranch !== 'develop') throw new Error('Expected site configuration isBeta boolean and canonicalBranch develop');
}

function validateCatalog(catalog, file) {
  if (!CATALOG_FILE_PATTERN.test(file) || !object(catalog) || catalog.id !== file.slice(14, -5)
    || !Array.isArray(catalog.tests) || !object(catalog.configurations) || Object.keys(catalog.configurations).length === 0) {
    throw new Error(`Test catalog ${file} does not match the schema-2 catalog contract`);
  }
  const ids = catalog.tests.map((test) => test?.id);
  if (new Set(ids).size !== ids.length || catalog.tests.some((test) => !hasText(test?.id) || !hasText(test.suite) || !hasText(test.name)
    || !object(test.problem) || Object.entries(test.problem).some(([key, value]) => !hasText(key) || !scalar(value)))) {
    throw new Error(`Test catalog ${file} contains an invalid or duplicate test definition`);
  }
  for (const [key, members] of Object.entries(catalog.configurations)) {
    if (!CONFIGURATION.test(key) || !Array.isArray(members) || members.length === 0 || new Set(members).size !== members.length
      || members.some((id) => !ids.includes(id))) throw new Error(`Test catalog ${file} contains an invalid configuration ${key}`);
  }
  return catalog;
}

export function validatePublishedResult(result) {
  if (!object(result) || !hasText(result.testId)) throw new Error('Result must contain a testId');
  const removed = ['exitCode', 'findings'].find((key) => Object.hasOwn(result, key));
  if (removed) throw new Error(`Result ${result.testId} contains removed field ${removed}`);
  if (!['completed', 'failed', 'timeout'].includes(result.status)) throw new Error(`Result ${result.testId} has invalid status`);
  if (!Object.hasOwn(result, 'durationSeconds') || !Object.hasOwn(result, 'error')) throw new Error(`Result ${result.testId} must contain durationSeconds and error`);
  if (result.status === 'completed') {
    if (!Number.isFinite(result.durationSeconds) || result.durationSeconds < 0) throw new Error(`Completed result ${result.testId} must contain nonnegative finite durationSeconds`);
    if (result.error !== null) throw new Error(`Completed result ${result.testId} cannot contain an error`);
  } else if (result.durationSeconds !== null) throw new Error(`${result.status} result ${result.testId} must have a null durationSeconds`);
  if (result.error !== null && !hasText(result.error)) throw new Error(`Result ${result.testId} error must be a non-empty string or null`);
  return result;
}

function normalizeRun(run, catalog, generatedAt) {
  const source = run?.source; const execution = run?.execution;
  const environment = run?.environment;
  if (typeof run?.id !== 'string' || !TOKEN.test(run.id)
    || !hasText(source?.branch) || typeof source.commit !== 'string' || !SHA.test(source.commit ?? '') || !isIsoTimestamp(source.committedAt)
    || (Object.hasOwn(source, 'message') && !hasText(source.message))
    || !isIsoTimestamp(execution?.completedAt) || !['auto', 'manual'].includes(execution.trigger) || !hasText(execution.machine)
    || !Array.isArray(environment) || environment.some((detail) => !hasText(detail?.key) || !hasText(detail?.label) || !scalar(detail?.value))
    || new Set(environment.map(({ key }) => key)).size !== environment.length || !Array.isArray(run.configurations) || run.configurations.length === 0) {
    throw new Error(`Run ${run?.id ?? '(unknown)'} does not match the schema-2 run contract`);
  }
  if (Date.parse(source.committedAt) > Date.parse(execution.completedAt) || Date.parse(execution.completedAt) > Date.parse(generatedAt)) throw new Error(`Run ${run.id} must be committed before completion and completed by publication`);
  if (Object.hasOwn(source, 'base') && (!hasText(source.base?.branch) || typeof source.base?.commit !== 'string' || !SHA.test(source.base?.commit ?? ''))) throw new Error(`Run ${run.id} has an invalid source base`);
  if (Object.hasOwn(source, 'pullRequest') && (!Number.isInteger(source.pullRequest?.number) || source.pullRequest.number <= 0
    || (Object.hasOwn(source.pullRequest, 'url') && (!safeUrl(source.pullRequest.url, true) || !new URL(source.pullRequest.url).pathname.endsWith(`/pull/${source.pullRequest.number}`))))) throw new Error(`Run ${run.id} has an invalid pullRequest`);

  const definitions = new Map(catalog.tests.map((test) => [test.id, test]));
  const seen = new Set();
  const tests = run.configurations.flatMap((configuration) => {
    const { target, threadingMode, results } = configuration;
    if (typeof threadingMode !== 'string' || !['ST', 'MT'].includes(threadingMode)) throw new Error(`Run ${run.id} has an invalid configuration threadingMode`);
    const key = `${target}:${threadingMode}`;
    if (!hasText(target) || !CONFIGURATION.test(key) || !Object.hasOwn(catalog.configurations, key) || seen.has(key) || !Array.isArray(results)) throw new Error(`Run ${run.id} has an invalid or duplicate configuration ${key}`);

    seen.add(key);
    results.forEach(validatePublishedResult);
    const ids = results.map(({ testId }) => testId);
    if (new Set(ids).size !== ids.length || !sameSet(ids, catalog.configurations[key])) throw new Error(`Run ${run.id} must contain exactly one result per ${key} catalog workload`);
    return results.map(({ testId, status, durationSeconds, error }) => ({ ...definitions.get(testId), status, durationSeconds, error,
      testId: `${key}:${testId}`, logicalTestId: testId, target, mode: threadingMode }));
  });
  return {
    runId: run.id, testCatalog: run.testCatalog, catalogId: catalog.id,
    timestamp: execution.completedAt, commitTimestamp: source.committedAt, trigger: execution.trigger, machineId: execution.machine,
    targets: [...new Set(run.configurations.map(({ target }) => target))], modes: ['ST', 'MT'].filter((mode) => run.configurations.some((c) => c.threadingMode === mode)),
    branch: source.branch, ...(source.base ? { sourceBase: { ...source.base } } : {}), ...(source.pullRequest ? { pullRequest: { ...source.pullRequest } } : {}),
    environmentId: environmentIdentity(environment),
    provenance: { rocjitsuCommitSha: source.commit, ...(source.message ? { commitMessage: source.message } : {}), details: environment },
    configurations: run.configurations.map(({ target, threadingMode }) => ({ target, threadingMode })), tests,
  };
}

function buildDashboardData(raw) {
  if (raw?.schemaVersion === 1) throw new Error('Dashboard schema 1 requires migration to schema 2');
  const sourceRuns = raw?.allRuns ?? raw?.runs;
  if (raw?.schemaVersion !== 2 || !Array.isArray(sourceRuns) || !Array.isArray(raw.testCatalog)) throw new Error('Expected schema-2 normalized data with runs and testCatalog');
  for (const run of sourceRuns) {
    if (!hasText(run?.runId) || !Array.isArray(run.tests) || !Array.isArray(run.configurations) || !isIsoTimestamp(run.timestamp)
      || !isIsoTimestamp(run.commitTimestamp) || !hasText(run.branch)) throw new Error('Invalid normalized dashboard run');
  }
  const allRuns = [...sourceRuns].sort(compareRunExecution);
  const runs = allRuns.filter(({ branch }) => branch === (raw.canonicalBranch ?? 'develop'));
  return { ...raw, canonicalBranch: raw.canonicalBranch ?? 'develop', allRuns, runs,
    targets: [...new Set(allRuns.flatMap(({ targets }) => targets))].sort(), suites: [...new Set(raw.testCatalog.map(({ suite }) => suite))].sort(), modes: ['ST', 'MT'],
    latestRun: runs.at(-1) ?? null, latestCommitRun: sortRunsByCommit(runs).at(-1) ?? null, backfillRunIds: backfillRunIds(runs) };
}

// Rehydrate processed normalized JSON through the same wire validator, not a schema bypass.
export function loadDashboardData(raw) {
  const shape = buildDashboardData(raw);
  const seenIds = new Set();
  for (const run of shape.allRuns) {
    if (seenIds.has(run.runId)) throw new Error(`Duplicate run ID ${run.runId}`);
    seenIds.add(run.runId);
  }
  const catalogs = { ...(raw.catalogs ?? {}) };
  const runs = shape.allRuns.map((run) => {
    for (const test of run.tests) {
      if (test.testId !== `${test.target}:${test.mode}:${test.logicalTestId}`
        || !run.configurations.some((c) => c.target === test.target && c.threadingMode === test.mode)) {
        throw new Error(`Run ${run.runId} has an invalid normalized test identity`);
      }
    }
    return { schemaVersion: CURRENT_SCHEMA_VERSION, id: run.runId, testCatalog: run.testCatalog,
      source: { branch: run.branch, commit: run.provenance?.rocjitsuCommitSha, committedAt: run.commitTimestamp,
        ...(Object.hasOwn(run.provenance ?? {}, 'commitMessage') ? { message: run.provenance.commitMessage } : {}),
        ...(run.sourceBase ? { base: run.sourceBase } : {}), ...(run.pullRequest ? { pullRequest: run.pullRequest } : {}) },
      execution: { completedAt: run.timestamp, trigger: run.trigger, machine: run.machineId },
      environment: run.provenance?.details ?? [],
      configurations: run.configurations.map(({ target, threadingMode }) => ({ target, threadingMode,
        results: run.tests.filter((t) => t.target === target && t.mode === threadingMode)
          .map(({ logicalTestId, status, durationSeconds, error }) => ({ testId: logicalTestId, status, durationSeconds, error })) })),
    };
  });
  const siteConfig = { repository: raw.repository, isBeta: raw.isBeta, canonicalBranch: shape.canonicalBranch };
  return validatePublishedDashboardData({ siteConfig, index: { generatedAt: raw.generatedAt, runFiles: runs.map((run) => publishedRunPath(run, siteConfig.canonicalBranch)) }, runs, catalogs }).data;
}

export function validatePublishedDashboardData({ index, runs, runErrors = [], catalogs = {}, catalogErrors = {}, siteConfig = DASHBOARD_SITE_CONFIG }) {
  validatePublishedManifest(index);
  validateDashboardSiteConfig(siteConfig);
  if (!Array.isArray(runs) || runs.length !== index.runFiles.length) throw new Error('Loaded runs must match index runFiles');
  const normalizedCatalogs = new Map(); const sourceCatalogs = new Map();
  const normalizedRuns = []; const failures = []; const sourceIds = new Set();
  index.runFiles.forEach((file, indexPosition) => {
    try {
      if (runErrors[indexPosition]) throw runErrors[indexPosition];
      const sourceRun = runs[indexPosition];
      if (typeof sourceRun?.id !== 'string' || !TOKEN.test(sourceRun.id)) throw new Error('Run does not match the schema-2 run contract: invalid ID');
      if (sourceIds.has(sourceRun.id)) throw new Error(`Duplicate run ID ${sourceRun.id}`);
      sourceIds.add(sourceRun.id);
      if (file.split('/').at(-1) !== `${sourceRun.id}.json`) throw new Error(`Run ${sourceRun.id} filename must match its ID`);
      const catalogFile = sourceRun.testCatalog;
      if (typeof catalogFile === 'string' && CATALOG_FILE_PATTERN.test(catalogFile) && Object.hasOwn(catalogs, catalogFile)) {
        sourceCatalogs.set(catalogFile, catalogs[catalogFile]);
      }
      if (isExcludedPluginRun(sourceRun)) return;
      const version = runSchemaVersion(sourceRun);
      if (file !== publishedRunPath(sourceRun, siteConfig.canonicalBranch)
        && !(version === 1 && file === `runs/${sourceRun.id}.json`)) {
        throw new Error(`Run ${sourceRun.id} filename must match its ID and branch directory`);
      }
      if (typeof catalogFile !== 'string' || !CATALOG_FILE_PATTERN.test(catalogFile)) throw new Error(`Run ${sourceRun.id} references an invalid test catalog`);
      if (catalogErrors[catalogFile]) throw catalogErrors[catalogFile];
      const migrated = version === 1 ? migrateSchema1Run(sourceRun, catalogs[catalogFile])
        : { run: sourceRun, catalog: catalogs[catalogFile] };
      if (migrated.run.schemaVersion !== CURRENT_RUN_SCHEMA_VERSION) {
        throw new Error(`Run ${sourceRun.id} has no migration from schema ${version} to requested schema ${CURRENT_RUN_SCHEMA_VERSION}`);
      }
      const catalog = validateCatalog(migrated.catalog, catalogFile);
      normalizedCatalogs.set(catalogFile, catalog);
      normalizedRuns.push(normalizeRun(migrated.run, catalog, index.generatedAt));
    } catch (error) { failures.push(`- ${file}: ${error.message}`); }
  });
  if (failures.length) throw new Error(`Dashboard data failed validation:\n${failures.join('\n')}`);
  const ids = new Set(); const commits = new Map(); const definitions = new Map();
  for (const run of normalizedRuns) {
    if (ids.has(run.runId)) throw new Error(`Duplicate run ID ${run.runId}`);
    ids.add(run.runId);
    const sha = run.provenance.rocjitsuCommitSha; const timestamp = Date.parse(run.commitTimestamp);
    if (commits.has(sha) && commits.get(sha) !== timestamp) throw new Error(`Commit ${sha} has conflicting committedAt values`);
    commits.set(sha, timestamp);
  }
  for (const [file, catalog] of normalizedCatalogs) for (const definition of catalog.tests) {
    const previous = definitions.get(definition.id);
    if (previous && definitionIdentity(previous) !== definitionIdentity(definition)) throw new Error(`Test ${definition.id} is defined differently in ${file}; publish a new test ID`);
    definitions.set(definition.id, definition);
  }
  const sourceData = { index, catalogs: Object.fromEntries(sourceCatalogs), runs };
  const data = buildDashboardData({ repository: siteConfig.repository, isBeta: siteConfig.isBeta, canonicalBranch: siteConfig.canonicalBranch, schemaVersion: CURRENT_SCHEMA_VERSION, generatedAt: index.generatedAt, testCatalog: [...definitions.values()], catalogs: Object.fromEntries(normalizedCatalogs), runs: normalizedRuns });
  return { data, sourceData };
}
