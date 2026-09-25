import { backfillRunIds, compareRunExecution, sortRunsByCommit } from './runOrdering.js';

const CURRENT_SCHEMA_VERSION = 1;
export const RUN_FILE_PATTERN = /^runs\/[A-Za-z0-9._-]+\.json$/;
export const CATALOG_FILE_PATTERN = /^test-catalogs\/[A-Za-z0-9._-]+\.json$/;
const COMMIT_SHA_PATTERN = /^[0-9a-f]{40}$/i;
const ISO_TIMESTAMP_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/;
const RUN_STATUSES = new Set(['completed', 'failed', 'timeout']);

function hasText(value) {
  return typeof value === 'string' && Boolean(value.trim());
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isIsoTimestamp(value) {
  return hasText(value)
    && ISO_TIMESTAMP_PATTERN.test(value)
    && Number.isFinite(Date.parse(value));
}

function isScalarValue(value) {
  return typeof value === 'string'
    || typeof value === 'number' && Number.isFinite(value)
    || typeof value === 'boolean';
}

function isEnvironmentValue(value) {
  return typeof value === 'string'
    ? hasText(value)
    : typeof value === 'number' ? Number.isFinite(value) : typeof value === 'boolean';
}

function isEnvironmentDetail(detail) {
  return hasText(detail?.key)
    && hasText(detail?.label)
    && isEnvironmentValue(detail?.value);
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
      hasText(key) && isScalarValue(value)
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
  const catalogIdFromPath = catalogPath.slice('test-catalogs/'.length, -'.json'.length);
  if (catalog.id !== catalogIdFromPath) {
    throw new Error(`Test catalog ${catalogPath} must contain id ${catalogIdFromPath}`);
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
  const referencedDefinitionIds = new Set(Object.values(catalog.targets).flat());
  const unreferencedDefinition = catalog.tests.find((definition) => (
    !referencedDefinitionIds.has(definition.id)
  ));
  if (unreferencedDefinition) {
    throw new Error(
      `Test catalog ${catalogPath} defines ${unreferencedDefinition.id} without assigning it to a target`,
    );
  }

  return catalog;
}

export function validatePublishedResult(result) {
  if (!isPlainObject(result) || !hasText(result.testId)) throw new Error('Result must contain a testId');
  const removedField = ['exitCode', 'findings'].find((field) => Object.hasOwn(result, field));
  if (removedField) throw new Error(`Result ${result.testId} contains removed field ${removedField}`);
  if (!RUN_STATUSES.has(result.status)) {
    throw new Error(`Result ${result.testId} has invalid status ${String(result.status)}`);
  }
  if (!Object.hasOwn(result, 'durationSeconds')) {
    throw new Error(`Result ${result.testId} must contain durationSeconds`);
  }
  if (!Object.hasOwn(result, 'error')) {
    throw new Error(`Result ${result.testId} must contain error`);
  }
  if (result.status === 'completed') {
    if (!Number.isFinite(result.durationSeconds) || result.durationSeconds <= 0) {
      throw new Error(`Completed result ${result.testId} must contain a positive durationSeconds`);
    }
    if (result.error != null) {
      throw new Error(`Completed result ${result.testId} cannot contain an error`);
    }
  } else if (result.durationSeconds != null) {
    throw new Error(`${result.status} result ${result.testId} must have a null durationSeconds`);
  }
  if (result.error != null && !hasText(result.error)) {
    throw new Error(`Result ${result.testId} error must be a non-empty string or null`);
  }
  return result;
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
    || !hasText(source.commit)
    || !COMMIT_SHA_PATTERN.test(source.commit)
    || !isIsoTimestamp(source.committedAt)
    || (Object.hasOwn(source, 'message') && !hasText(source.message))
    || !isIsoTimestamp(execution.completedAt)
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
    try {
      targetGroup.results.forEach(validatePublishedResult);
    } catch (error) {
      throw new Error(`Run ${run.id} has an invalid ${targetGroup.id} result: ${error.message}`, { cause: error });
    }
    if (
      new Set(resultIds).size !== resultIds.length
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
        error: result.error ?? null,
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
    environmentId: run.environmentId,
    targets: [...run.targets].sort(),
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
    || !isIsoTimestamp(run.timestamp)
    || !isIsoTimestamp(run.commitTimestamp)
    || !hasText(run.branch)
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
  const targets = [...new Set(runs.flatMap((run) => run.targets))];

  return {
    ...raw,
    pluginRuns,
    runs,
    latestRun: runs.at(-1) ?? null,
    latestCommitRun,
    backfillRunIds: backfillRunIds(runs),
    targets,
    suites: [...new Set(raw.testCatalog.map((test) => test.suite))].sort(),
  };
}

export function loadDashboardData(raw) {
  return buildDashboardData(raw);
}

export function validatePublicationPolicy(runs) {
  if (!Array.isArray(runs)) throw new Error('Expected published runs to be an array');
  const referenceRun = runs[0];
  if (!referenceRun) return [];

  return runs.flatMap((run) => {
    const issues = [];
    if (run.source.branch !== 'develop') {
      issues.push({
        runId: run.id,
        message: `Run ${run.id} must use source branch develop`,
      });
    }
    if (run.execution.machine !== referenceRun.execution.machine) {
      issues.push({
        runId: run.id,
        message: `Run ${run.id} uses machine ${run.execution.machine}; `
          + `expected ${referenceRun.execution.machine}`,
      });
    }
    return issues;
  });
}

export function validatePublishedDashboardData({
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
  let repositoryUrl;
  try {
    repositoryUrl = new URL(metadata.repository);
  } catch {
    throw new Error('Expected dashboard metadata to contain an HTTP repository URL');
  }
  if (!['http:', 'https:'].includes(repositoryUrl.protocol)) {
    throw new Error('Expected dashboard metadata to contain an HTTP repository URL');
  }
  if (typeof metadata.isBeta !== 'boolean') {
    throw new Error('Expected dashboard metadata to contain an isBeta boolean');
  }
  if (!index || !isIsoTimestamp(index.generatedAt) || !Array.isArray(index.runFiles)) {
    throw new Error('Expected the dashboard data index to contain generatedAt and runFiles');
  }
  if (!Array.isArray(runs)) {
    throw new Error('Expected loaded runs to be an array');
  }

  const normalizedRuns = [];
  const acceptedSourceRuns = [];
  const validationFailures = [];
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
      if (hasText(publishedRun?.id) && runFile !== `runs/${publishedRun.id}.json`) {
        throw new Error(`Run ${publishedRun.id} must be published as runs/${publishedRun.id}.json`);
      }
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
      validationFailures.push({
        runFile: typeof runFile === 'string' ? runFile : String(runFile),
        message: runError instanceof Error ? runError.message : String(runError),
      });
    }
  });
  if (validationFailures.length > 0) {
    throw new Error(`Dashboard data failed validation:\n${validationFailures
      .map(({ runFile, message }) => `- ${runFile}: ${message}`)
      .join('\n')}`);
  }

  for (const [comparisonId, group] of comparisonGroups) {
    const hasInstrumentedPlugin = [...group.plugins].some((pluginId) => pluginId !== 'vanilla');
    if (hasInstrumentedPlugin && !group.plugins.has('vanilla')) {
      throw new Error(`Comparison ${comparisonId} has plugin runs without a Vanilla baseline`);
    }
  }

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
    runs: normalizedRuns,
  });

  if (!data.latestRun) throw new Error('The data files do not contain any Vanilla benchmark runs');

  const publicationIssues = validatePublicationPolicy(acceptedSourceRuns);
  if (publicationIssues.length > 0) {
    throw new Error(`Dashboard data failed publication policy:\n${publicationIssues
      .map(({ runId, message }) => `- ${runId}: ${message}`)
      .join('\n')}`);
  }
  return { data, sourceData };
}
