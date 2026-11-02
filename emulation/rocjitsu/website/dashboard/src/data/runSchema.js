export const CURRENT_RUN_SCHEMA_VERSION = 1;
export const THREAD_POLICIES = Object.freeze(['single', 'default']);
export const configurationKey = (target, policy) => `${target}/${policy}`;
export const hasConfiguration = (run, target, policy) => (run?.configurations ?? [])
  .some((configuration) => configuration.target === target && configuration.thread_policy === policy);

export function publishedRunPath(run, canonicalBranch = 'develop') {
  const directory = run.publication?.source?.branch === canonicalBranch ? 'default-branch' : 'side-branches';
  return `runs/${directory}/${run.publication?.id}.json`;
}

export function runSchemaVersion(run) {
  if (run === null || typeof run !== 'object' || Array.isArray(run)) throw new Error('Run must be an object');
  if (run.schema_version !== CURRENT_RUN_SCHEMA_VERSION) {
    throw new Error(`Run ${run.publication?.id ?? '(unknown)'} has unsupported schema version ${run.schema_version ?? '(missing)'}; expected corpus schema 1`);
  }
  return run.schema_version;
}
