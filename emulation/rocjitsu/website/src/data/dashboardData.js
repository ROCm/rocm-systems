import { backfillRunIds, compareRunExecution, isRunCompleted, sortRunsByCommit } from './runOrdering';

const CURRENT_SCHEMA_VERSION = 1;
const RUN_FILE_PATTERN = /^runs\/[A-Za-z0-9._-]+\.json$/;
const CATALOG_FILE_PATTERN = /^test-catalogs\/[A-Za-z0-9._-]+\.json$/;
const RUN_STATUSES = new Set(['completed', 'failed', 'timeout']);

function hasText(value) {
  return typeof value === 'string' && Boolean(value.trim());
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isEnvironmentDetail(detail) {
  return hasText(detail?.key)
    && hasText(detail?.label)
    && ['string', 'number', 'boolean'].includes(typeof detail?.value);
}

function environmentIdentity(environment) {
  return JSON.stringify(environment
    .map(({ key, value }) => [key, value])
    .sort(([left], [right]) => left.localeCompare(right)));
}

function isTestDefinition(definition) {
  return hasText(definition?.id)
    && hasText(definition?.suite)
    && hasText(definition?.name)
    && isPlainObject(definition.problem)
    && Object.entries(definition.problem).every(([key, value]) => (
      hasText(key) && ['string', 'number', 'boolean'].includes(typeof value)
    ));
}

function normalizeCatalog(catalog, catalogPath) {
  if (
    !isPlainObject(catalog)
    || !hasText(catalog.id)
    || !Array.isArray(catalog.tests)
    || catalog.tests.length === 0
    || !isPlainObject(catalog.targets)
    || Object.keys(catalog.targets).length === 0
  ) {
    throw new Error(`Test catalog ${catalogPath} does not match the schema-version-${CURRENT_SCHEMA_VERSION} contract`);
  }

  const definitionIds = catalog.tests.map((definition) => definition?.id);
  if (
    catalog.tests.some((definition) => !isTestDefinition(definition))
    || new Set(definitionIds).size !== definitionIds.length
  ) {
    throw new Error(`Test catalog ${catalogPath} contains an invalid or duplicate test definition`);
  }

  const definitionIdSet = new Set(definitionIds);
  for (const [target, testIds] of Object.entries(catalog.targets)) {
    if (
      !hasText(target)
      || !Array.isArray(testIds)
      || testIds.length === 0
      || testIds.some((testId) => !hasText(testId) || !definitionIdSet.has(testId))
      || new Set(testIds).size !== testIds.length
    ) {
      throw new Error(`Test catalog ${catalogPath} contains an invalid test set for ${target || '(unknown target)'}`);
    }
  }

  return catalog;
}

function isFinding(finding) {
  return isPlainObject(finding)
    && hasText(finding.type)
    && hasText(finding.summary)
    && (!Object.hasOwn(finding, 'location') || hasText(finding.location));
}

function isPublishedResult(result) {
  return hasText(result?.testId)
    && RUN_STATUSES.has(result.status)
    && (result.status === 'completed'
      ? Number.isFinite(result.durationSeconds)
      : result.durationSeconds == null)
    && (!Object.hasOwn(result, 'findings')
      || (Array.isArray(result.findings) && result.findings.every(isFinding)));
}

function sameStringSet(left, right) {
  const sortedLeft = [...left].sort();
  const sortedRight = [...right].sort();
  return sortedLeft.length === sortedRight.length
    && sortedLeft.every((value, index) => value === sortedRight[index]);
}

function normalizePublishedRun(run, catalog) {
  const source = run?.source;
  const execution = run?.execution;
  const environment = run?.environment;
  const plugin = run?.plugin;
  const targetGroups = run?.targets;
  const runLabel = hasText(run?.id) ? run.id : '(unknown)';

  if (
    !hasText(run?.id)
    || !hasText(run?.comparisonId)
    || !hasText(run?.testCatalog)
    || !plugin
    || !hasText(plugin.id)
    || !hasText(plugin.name)
    || (Object.hasOwn(plugin, 'version') && !hasText(plugin.version))
    || (Object.hasOwn(plugin, 'options') && !isPlainObject(plugin.options))
    || !source
    || !execution
    || !hasText(source.branch)
    || source.branch !== 'develop'
    || !hasText(source.commit)
    || !Number.isFinite(Date.parse(source.committedAt))
    || !Number.isFinite(Date.parse(execution.completedAt))
    || !['auto', 'manual'].includes(execution.trigger)
    || !hasText(execution.machine)
    || !Array.isArray(environment)
    || environment.some((detail) => !isEnvironmentDetail(detail))
    || new Set(environment.map((detail) => detail.key)).size !== environment.length
    || !Array.isArray(targetGroups)
    || targetGroups.length === 0
  ) {
    throw new Error(`Run ${runLabel} does not match the schema-version-${CURRENT_SCHEMA_VERSION} run contract`);
  }

  const targetIds = targetGroups.map((targetGroup) => targetGroup?.id);
  const catalogTargetIds = Object.keys(catalog.targets);
  if (
    targetGroups.some((targetGroup) => !hasText(targetGroup?.id) || !Array.isArray(targetGroup.results))
    || new Set(targetIds).size !== targetIds.length
    || !sameStringSet(targetIds, catalogTargetIds)
  ) {
    throw new Error(`Run ${run.id} does not contain exactly the targets required by ${catalog.id}`);
  }

  const definitions = new Map(catalog.tests.map((definition) => [definition.id, definition]));
  const tests = targetGroups.flatMap((targetGroup) => {
    const expectedIds = catalog.targets[targetGroup.id];
    const resultIds = targetGroup.results.map((result) => result?.testId);
    if (
      targetGroup.results.some((result) => !isPublishedResult(result))
      || new Set(resultIds).size !== resultIds.length
      || !sameStringSet(resultIds, expectedIds)
    ) {
      throw new Error(`Run ${run.id} does not contain exactly one valid result for every ${targetGroup.id} catalog test`);
    }

    return targetGroup.results.map((result) => {
      const definition = definitions.get(result.testId);
      return {
        ...definition,
        testId: `${targetGroup.id}:${result.testId}`,
        logicalTestId: result.testId,
        target: targetGroup.id,
        durationSeconds: result.durationSeconds ?? null,
        status: result.status,
        exitCode: result.exitCode ?? null,
        timedOut: result.status === 'timeout',
        error: result.error ?? null,
        findings: result.findings ?? [],
      };
    });
  });

  return {
    runId: run.id,
    comparisonId: run.comparisonId,
    testCatalog: run.testCatalog,
    catalogId: catalog.id,
    plugin: { ...plugin },
    timestamp: execution.completedAt,
    commitTimestamp: source.committedAt,
    trigger: execution.trigger,
    machineId: execution.machine,
    targets: targetIds,
    branch: source.branch,
    environmentId: environmentIdentity(environment),
    provenance: {
      rocjitsuCommitSha: source.commit,
      ...(hasText(source.message) ? { commitMessage: source.message } : {}),
      details: environment,
    },
    tests,
  };
}

function testDefinitionIdentity(definition) {
  return JSON.stringify([
    definition.suite,
    definition.name,
    Object.entries(definition.problem ?? {}).sort(([left], [right]) => left.localeCompare(right)),
  ]);
}

function comparisonIdentity(run) {
  return JSON.stringify({
    testCatalog: run.testCatalog,
    branch: run.branch,
    commitTimestamp: run.commitTimestamp,
    trigger: run.trigger,
    machineId: run.machineId,
    targets: [...run.targets].sort(),
    environmentId: run.environmentId,
    commit: run.provenance.rocjitsuCommitSha,
    message: run.provenance.commitMessage ?? null,
  });
}

function buildDashboardData(raw) {
  const allRuns = raw?.pluginRuns ?? raw?.runs;
  if (
    !raw
    || raw.schemaVersion !== CURRENT_SCHEMA_VERSION
    || !Array.isArray(allRuns)
    || !Array.isArray(raw.testCatalog)
  ) {
    throw new Error(`Expected schema-version-${CURRENT_SCHEMA_VERSION} dashboard data with runs and testCatalog arrays`);
  }

  const definitionIds = raw.testCatalog.map((definition) => definition?.id);
  if (
    raw.testCatalog.some((definition) => !isTestDefinition(definition))
    || new Set(definitionIds).size !== definitionIds.length
  ) {
    throw new Error('The test catalog contains an invalid or duplicate benchmark definition');
  }

  const invalidRun = allRuns.find((run) => (
    !hasText(run?.runId)
    || !hasText(run?.comparisonId)
    || !hasText(run?.plugin?.id)
    || !Number.isFinite(Date.parse(run.timestamp))
    || !Number.isFinite(Date.parse(run.commitTimestamp))
    || run.branch !== 'develop'
    || !['auto', 'manual'].includes(run.trigger)
    || !hasText(run.machineId)
    || !Array.isArray(run.targets)
    || !Array.isArray(run.tests)
    || !hasText(run.environmentId)
    || !hasText(run.provenance?.rocjitsuCommitSha)
  ));
  if (invalidRun) {
    throw new Error(`Run ${invalidRun.runId ?? '(unknown)'} is not a valid official develop run`);
  }

  const commitTimestamps = new Map();
  const inconsistentCommit = allRuns.find((run) => {
    const sha = run.provenance.rocjitsuCommitSha;
    const timestamp = Date.parse(run.commitTimestamp);
    const existingTimestamp = commitTimestamps.get(sha);
    if (existingTimestamp !== undefined && existingTimestamp !== timestamp) return true;
    commitTimestamps.set(sha, timestamp);
    return false;
  });
  if (inconsistentCommit) {
    throw new Error(
      `Commit ${inconsistentCommit.provenance.rocjitsuCommitSha} has conflicting committedAt values`,
    );
  }

  const pluginRuns = [...allRuns].sort(compareRunExecution);
  const runs = pluginRuns.filter((run) => run.plugin.id === 'vanilla');
  const latestCommitRun = sortRunsByCommit(runs).at(-1) ?? null;
  const latestCompletedRun = sortRunsByCommit(runs.filter(isRunCompleted)).at(-1) ?? null;
  const targets = [...new Set(runs.flatMap((run) => run.targets))];
  const plugins = [...new Map(pluginRuns.map((run) => [run.plugin.id, run.plugin])).values()];

  return {
    ...raw,
    pluginRuns,
    runs,
    latestRun: runs.at(-1) ?? null,
    latestCommitRun,
    latestCompletedRun,
    backfillRunIds: backfillRunIds(runs),
    targets,
    plugins,
    suites: [...new Set(raw.testCatalog.map((test) => test.suite))].sort(),
  };
}

export function loadDashboardData(raw) {
  return buildDashboardData(raw);
}

export function loadPublishedDashboardData({
  metadata,
  index,
  runs,
  runErrors = [],
  catalogs = {},
  catalogErrors = {},
}) {
  if (metadata?.schemaVersion !== CURRENT_SCHEMA_VERSION) {
    throw new Error(`Unsupported dashboard schema version ${metadata?.schemaVersion ?? '(missing)'}`);
  }
  if (typeof metadata.isBeta !== 'boolean') {
    throw new Error('Expected dashboard metadata to contain an isBeta boolean');
  }
  if (!index || !Number.isFinite(Date.parse(index.generatedAt)) || !Array.isArray(index.runFiles)) {
    throw new Error('Expected the dashboard data index to contain generatedAt and runFiles');
  }
  if (!Array.isArray(runs)) {
    throw new Error('Expected loaded runs to be an array');
  }

  const normalizedRuns = [];
  const acceptedSourceRuns = [];
  const warnings = [];
  const seenRunFiles = new Set();
  const seenRunIds = new Set();
  const commitTimestamps = new Map();
  const comparisonGroups = new Map();
  const normalizedCatalogs = new Map();

  index.runFiles.forEach((runFile, runIndex) => {
    const validRunFile = typeof runFile === 'string' && RUN_FILE_PATTERN.test(runFile);
    let error = runErrors[runIndex] ?? null;
    if (!validRunFile) error ??= new Error(`Invalid run filename: ${String(runFile)}`);
    if (validRunFile && seenRunFiles.has(runFile)) error ??= new Error('Duplicate run filename');
    if (validRunFile) seenRunFiles.add(runFile);

    try {
      if (error) throw error;
      const publishedRun = runs[runIndex];
      const catalogPath = publishedRun?.testCatalog;
      if (!hasText(catalogPath) || !CATALOG_FILE_PATTERN.test(catalogPath)) {
        throw new Error(`Run ${publishedRun?.id ?? '(unknown)'} references an invalid test catalog`);
      }
      if (catalogErrors[catalogPath]) throw catalogErrors[catalogPath];
      let catalog = normalizedCatalogs.get(catalogPath);
      if (!catalog) {
        if (!catalogs[catalogPath]) throw new Error(`Unable to load test catalog ${catalogPath}`);
        catalog = normalizeCatalog(catalogs[catalogPath], catalogPath);
        normalizedCatalogs.set(catalogPath, catalog);
      }

      const normalizedRun = normalizePublishedRun(publishedRun, catalog);
      if (seenRunIds.has(normalizedRun.runId)) throw new Error(`Duplicate run ID ${normalizedRun.runId}`);
      const commitSha = normalizedRun.provenance.rocjitsuCommitSha;
      const commitTimestamp = Date.parse(normalizedRun.commitTimestamp);
      const existingCommitTimestamp = commitTimestamps.get(commitSha);
      if (existingCommitTimestamp !== undefined && existingCommitTimestamp !== commitTimestamp) {
        throw new Error(`Commit ${commitSha} has conflicting committedAt values`);
      }

      const existingGroup = comparisonGroups.get(normalizedRun.comparisonId);
      if (existingGroup) {
        if (existingGroup.identity !== comparisonIdentity(normalizedRun)) {
          throw new Error(`Run ${normalizedRun.runId} does not match comparison ${normalizedRun.comparisonId}`);
        }
        if (existingGroup.plugins.has(normalizedRun.plugin.id)) {
          throw new Error(`Comparison ${normalizedRun.comparisonId} repeats plugin ${normalizedRun.plugin.id}`);
        }
        existingGroup.plugins.add(normalizedRun.plugin.id);
      } else {
        comparisonGroups.set(normalizedRun.comparisonId, {
          identity: comparisonIdentity(normalizedRun),
          plugins: new Set([normalizedRun.plugin.id]),
        });
      }

      seenRunIds.add(normalizedRun.runId);
      commitTimestamps.set(commitSha, commitTimestamp);
      normalizedRuns.push(normalizedRun);
      acceptedSourceRuns.push(publishedRun);
    } catch (runError) {
      warnings.push({
        runFile: typeof runFile === 'string' ? runFile : String(runFile),
        message: runError instanceof Error ? runError.message : String(runError),
      });
    }
  });

  const derivedCatalog = new Map();
  const definitionSources = new Map();
  [...normalizedRuns]
    .sort(compareRunExecution)
    .forEach((run) => run.tests.forEach((test) => {
      const definition = {
        id: test.logicalTestId,
        suite: test.suite,
        name: test.name,
        problem: test.problem,
      };
      const identity = testDefinitionIdentity(definition);
      const source = definitionSources.get(definition.id);
      if (source && source.identity !== identity) {
        throw new Error(
          `Test ${definition.id} is defined differently by ${source.catalog} and ${run.testCatalog}; `
          + 'publish a new test ID whenever the suite, name, or problem changes',
        );
      }
      if (!source) definitionSources.set(definition.id, { identity, catalog: run.testCatalog });
      derivedCatalog.set(definition.id, definition);
    }));

  const sourceData = {
    metadata,
    index,
    catalogs: Object.fromEntries(normalizedCatalogs),
    runs: acceptedSourceRuns,
  };
  const data = buildDashboardData({
    ...metadata,
    generatedAt: index.generatedAt,
    testCatalog: [...derivedCatalog.values()],
    testCatalogs: [...normalizedCatalogs.values()],
    runs: normalizedRuns,
  });

  if (!data.latestRun) throw new Error('The data files do not contain any Vanilla benchmark runs');
  if (!data.latestCompletedRun) throw new Error('The data files do not contain a completed Vanilla benchmark run');

  return { data, sourceData, warnings };
}

// A dataset of several hundred runs is one HTTP request per run. Browsers queue beyond their own
// per-host limit anyway, and an unbounded fan-out makes every request share the same slow ramp, so
// the loader keeps a fixed number of requests in flight and reports progress as they settle.
export const MAX_CONCURRENT_RUN_REQUESTS = 8;
export const DEFAULT_REQUEST_TIMEOUT_MS = 20_000;
export const DEFAULT_LOAD_TIMEOUT_MS = 60_000;

const loadCancellation = Symbol('dashboard-load-cancellation');

class LoadCancelledError extends Error {
  constructor() {
    super('Dashboard data loading was cancelled');
    this.name = 'AbortError';
    this[loadCancellation] = true;
  }
}

export function isLoadCancelled(error) {
  return error?.[loadCancellation] === true;
}

function throwIfCancelled(signal) {
  if (signal?.aborted) throw new LoadCancelledError();
}

async function fetchJsonResource(url, {
  fetchImpl,
  signal,
  timeoutMs,
  cache,
  resourceType,
}) {
  throwIfCancelled(signal);
  const controller = new AbortController();
  const forwardAbort = () => controller.abort();
  signal?.addEventListener('abort', forwardAbort, { once: true });
  let timedOut = false;
  const timer = Number.isFinite(timeoutMs) && timeoutMs > 0
    ? setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs)
    : null;

  try {
    const response = await fetchImpl(String(url), {
      signal: controller.signal,
      ...(cache ? { cache } : {}),
    });
    if (!response.ok) {
      throw new Error(`Unable to load ${resourceType} ${url} (${response.status} ${response.statusText})`);
    }
    try {
      return await response.json();
    } catch (parseError) {
      throw new Error(`Unable to parse ${resourceType} ${url} as JSON: ${parseError.message}`, { cause: parseError });
    }
  } catch (error) {
    if (signal?.aborted) throw new LoadCancelledError();
    if (timedOut) throw new Error(`Timed out after ${timeoutMs} ms loading ${resourceType} ${url}`, { cause: error });
    throw error;
  } finally {
    if (timer) clearTimeout(timer);
    signal?.removeEventListener('abort', forwardAbort);
  }
}

async function mapWithConcurrency(items, limit, worker) {
  const results = new Array(items.length);
  let nextIndex = 0;
  const workerCount = Math.max(1, Math.min(limit, items.length));
  await Promise.all(Array.from({ length: workerCount }, async () => {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await worker(items[index], index);
    }
  }));
  return results;
}

export async function loadDashboardDataFiles({
  metadataUrl,
  indexUrl,
  onManifest,
  onProgress,
  signal,
  fetch: fetchImpl = globalThis.fetch.bind(globalThis),
  concurrency = MAX_CONCURRENT_RUN_REQUESTS,
  requestTimeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
  loadTimeoutMs = DEFAULT_LOAD_TIMEOUT_MS,
}) {
  const loadController = new AbortController();
  const forwardAbort = () => loadController.abort();
  if (signal?.aborted) forwardAbort();
  else signal?.addEventListener('abort', forwardAbort, { once: true });
  let loadTimedOut = false;
  const loadTimer = Number.isFinite(loadTimeoutMs) && loadTimeoutMs > 0
    ? setTimeout(() => {
      loadTimedOut = true;
      loadController.abort();
    }, loadTimeoutMs)
    : null;

  try {
  // metadata.json and index.json are the only mutable documents in the contract, so they must
  // revalidate on every load while immutable runs and catalogs use ordinary HTTP caching.
  const loadSignal = loadController.signal;
  const mutableRequest = { fetchImpl, signal: loadSignal, timeoutMs: requestTimeoutMs, cache: 'no-store' };
  const immutableRequest = { fetchImpl, signal: loadSignal, timeoutMs: requestTimeoutMs };
  const [metadata, index] = await Promise.all([
    fetchJsonResource(metadataUrl, { ...mutableRequest, resourceType: 'dashboard metadata' }),
    fetchJsonResource(indexUrl, { ...mutableRequest, resourceType: 'dashboard data index' }),
  ]);
  if (!index || !Array.isArray(index.runFiles)) {
    throw new Error('Expected the dashboard data index to contain a runFiles array');
  }
  onManifest?.({ ...metadata, generatedAt: index.generatedAt });

  const total = index.runFiles.length;
  let loaded = 0;
  onProgress?.({ loaded, total });

  const runResults = await mapWithConcurrency(index.runFiles, concurrency, async (runFile) => {
    const settle = (result) => {
      loaded += 1;
      onProgress?.({ loaded, total });
      return result;
    };
    if (typeof runFile !== 'string' || !RUN_FILE_PATTERN.test(runFile)) {
      return settle({ run: null, error: new Error(`Invalid run filename: ${String(runFile)}`) });
    }
    try {
      const run = await fetchJsonResource(new URL(runFile, indexUrl), {
        ...immutableRequest,
        resourceType: 'run file',
      });
      return settle({ run, error: null });
    } catch (error) {
      // Cancellation is a caller decision about the whole load, never one skippable run.
      if (isLoadCancelled(error)) throw error;
      return settle({ run: null, error });
    }
  });

  const catalogPaths = [...new Set(runResults
    .map((result) => result.run?.testCatalog)
    .filter((catalogPath) => hasText(catalogPath) && CATALOG_FILE_PATTERN.test(catalogPath)))];
  const catalogResults = await mapWithConcurrency(catalogPaths, concurrency, async (catalogPath) => {
    try {
      const catalog = await fetchJsonResource(new URL(catalogPath, indexUrl), {
        ...immutableRequest,
        resourceType: 'test catalog',
      });
      return { catalogPath, catalog, error: null };
    } catch (error) {
      if (isLoadCancelled(error)) throw error;
      return { catalogPath, catalog: null, error };
    }
  });

  throwIfCancelled(loadSignal);

  return loadPublishedDashboardData({
    metadata,
    index,
    runs: runResults.map((result) => result.run),
    runErrors: runResults.map((result) => result.error),
    catalogs: Object.fromEntries(catalogResults.map((result) => [result.catalogPath, result.catalog])),
    catalogErrors: Object.fromEntries(catalogResults
      .filter((result) => result.error)
      .map((result) => [result.catalogPath, result.error])),
  });
  } catch (error) {
    if (signal?.aborted) throw new LoadCancelledError();
    if (loadTimedOut && isLoadCancelled(error)) {
      throw new Error(`Timed out after ${loadTimeoutMs} ms loading dashboard data`, { cause: error });
    }
    throw error;
  } finally {
    if (loadTimer) clearTimeout(loadTimer);
    signal?.removeEventListener('abort', forwardAbort);
  }
}
