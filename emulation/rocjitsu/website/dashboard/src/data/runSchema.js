import metadata from '../config/metadata.json' with { type: 'json' };

// Metadata selects the run format the website wants to consume.
export const CURRENT_RUN_SCHEMA_VERSION = metadata.schemaVersion;
export const SUPPORTED_RUN_SCHEMA_VERSIONS = Object.freeze([1, 2]);
const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const text = (value) => typeof value === 'string' && Boolean(value.trim());

export function publishedRunPath(run, canonicalBranch = 'develop') {
  const directory = run.source?.branch === canonicalBranch ? 'default-branch' : 'side-branches';
  return `runs/${directory}/${run.id}.json`;
}

// Explicit product policy: instrumented runs are not ordinary measurements.
export function isExcludedPluginRun(run) {
  return text(run?.plugin?.id) && run.plugin.id !== 'vanilla';
}

export function runSchemaVersion(run) {
  if (!object(run)) throw new Error('Run must be an object');
  if (Object.hasOwn(run, 'plugin') && (!object(run.plugin) || !text(run.plugin.id))) {
    throw new Error(`Run ${run.id ?? '(unknown)'} has invalid plugin identity`);
  }
  // Existing runs without a version are schema 1; the migration validates their shape.
  const version = Object.hasOwn(run, 'schemaVersion') ? run.schemaVersion : 1;
  if (!SUPPORTED_RUN_SCHEMA_VERSIONS.includes(version)) {
    throw new Error(`Run ${run.id ?? '(unknown)'} has unsupported schema version ${version ?? '(missing)'}`);
  }
  if (version === 2 && Object.hasOwn(run, 'targets')) throw new Error(`Run ${run.id} has legacy targets in schema 2`);
  return version;
}

export function migrateSchema1Run(run, catalog) {
  if (!text(run.comparisonId) || !text(run.plugin?.name) || !Array.isArray(run.targets)
    || run.targets.length === 0 || Object.hasOwn(run, 'configurations')
    || !object(catalog?.targets) || Object.hasOwn(catalog, 'configurations')) {
    throw new Error(`Run ${run.id} does not match the schema-1 run/catalog contract`);
  }
  const targets = run.targets.map((group) => group?.id);
  const expected = Object.keys(catalog.targets);
  if (new Set(targets).size !== targets.length || targets.length !== expected.length
    || targets.some((target) => !text(target) || !Object.hasOwn(catalog.targets, target))) {
    throw new Error(`Run ${run.id} must contain exactly the schema-1 catalog targets`);
  }
  for (const group of run.targets) {
    if (!Array.isArray(group.results) || group.results.some((result) => result?.status === 'completed'
      && (!Number.isFinite(result.durationSeconds) || result.durationSeconds <= 0))) {
      throw new Error(`Run ${run.id} has invalid schema-1 results`);
    }
  }
  // Legacy measurements are MT
  return {
    run: {
      schemaVersion: 2, id: run.id, testCatalog: run.testCatalog,
      source: run.source, execution: run.execution, environment: run.environment,
      configurations: run.targets.map(({ id, results }) => ({ target: id, threadingMode: 'MT', results })),
    },
    catalog: {
      id: catalog.id, tests: catalog.tests,
      configurations: Object.fromEntries(Object.entries(catalog.targets).map(([target, ids]) => [`${target}:MT`, ids])),
    },
  };
}
