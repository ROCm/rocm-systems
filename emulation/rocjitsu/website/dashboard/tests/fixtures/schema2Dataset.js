// Fictional deterministic schema-2 measurements. Test/fixtures mode ONLY.
import { mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const CONFIGURATION_KEYS = ['gfx1250:ST', 'gfx1250:MT', 'gfx950:ST', 'gfx950:MT'];
const definitions = [
  ['a', 'Triton', 'Fictional GEMM', { operation: 'GEMM', m: 1024 }],
  ['b', 'Llama', 'Fictional decode', { tokens: 128 }],
  ['c', 'Triton', 'Fictional retired workload', { size: 64 }],
  ['d', 'Triton', 'Fictional added workload D', { size: 256 }],
  ['e', 'Llama', 'Fictional added workload E', { size: 512 }],
].map(([id, suite, name, problem]) => ({ id, suite, name, problem }));
const sha = (index) => `${index.toString(16).padStart(4, '0')}f1c7`.repeat(5);
const iso = (day, hour = 6, minute = 0) => new Date(Date.UTC(2026, 8, 11 + day, hour, minute)).toISOString();
const catalogFor = (current) => ({
  id: current ? 'fictional-current' : 'fictional-old',
  tests: structuredClone(definitions.filter(({ id }) => (current ? ['a', 'b', 'd', 'e'] : ['a', 'b', 'c']).includes(id))),
  configurations: Object.fromEntries(CONFIGURATION_KEYS.map((key) => [key,
    current ? (key === 'gfx950:MT' ? ['a', 'b', 'd'] : ['a', 'b', 'd', 'e']) : ['a', 'b', 'c'],
  ])),
});

export function createSchema2Publication() {
  const catalogs = Object.fromEntries([false, true].map((current) => {
    const catalog = catalogFor(current);
    return [`test-catalogs/${catalog.id}.json`, catalog];
  }));
  const canonical = Array.from({ length: 24 }, (_, index) => {
    const current = index >= 8;
    const testCatalog = `test-catalogs/fictional-${current ? 'current' : 'old'}.json`;
    const id = `fictional-develop-${String(index).padStart(2, '0')}`;
    return {
      id, testCatalog,
      source: { branch: 'develop', commit: sha(index + 1), committedAt: iso(index, 4), message: `FICTIONAL develop measurement ${index}` },
      execution: { completedAt: index === 21 ? '2026-10-05T10:00:00.000Z' : iso(index), trigger: index === 21 ? 'manual' : 'auto', machine: index < 16 ? 'fictional-node-a' : 'fictional-node-b' },
      environment: [{ key: 'sdk', label: 'Fictional SDK', value: index < 16 ? 'test-a' : 'test-b' }, { key: 'enabled', label: 'Feature enabled', value: true }, { key: 'cores', label: 'CPU cores', value: 32 }],
      configurations: CONFIGURATION_KEYS.filter((key) => !(index === 12 && key === 'gfx950:MT')).map((key, configIndex) => {
        const [target, mode] = key.split(':');
        return { target, threadingMode: mode,
          results: catalogs[testCatalog].configurations[key].map((testId, testIndex) => {
            const status = (index === 8 && mode === 'ST' && testId === 'd') || (index === 9 && mode === 'ST' && testId === 'e') ? 'failed'
              : index === 8 && testId === 'e' ? 'timeout' : 'completed';
            return { testId, status, durationSeconds: status !== 'completed' ? null
              : index === 14 && testId === 'a' ? 0 : Number((20 + testIndex * 10 + configIndex * 5 - index * 0.2).toFixed(2)), error: status === 'completed' ? null : 'Fictional diagnostic' };
          }) };
      }),
    };
  });
  const branches = Array.from({ length: 20 }, (_, index) => {
    const run = structuredClone(canonical[23]);
    run.id = `fictional-branch-${String(index + 1).padStart(2, '0')}`;
    run.source = { branch: `fictional/optimization-${String(index + 1).padStart(2, '0')}`, commit: sha(100 + index), committedAt: '2026-10-05T09:00:00.000Z', message: `FICTIONAL branch ${index + 1}`,
      ...(index % 2 === 0 ? { base: { branch: 'develop', commit: canonical[20].source.commit }, pullRequest: { number: 1000 + index } } : {}),
    };
    run.execution.completedAt = `2026-10-05T11:${String(index).padStart(2, '0')}:00.000Z`;
    run.execution.trigger = 'manual';
    run.execution.machine = 'fictional-branch-node';
    run.environment[0].value = 'test-branch';
    if (index === 1) run.configurations = run.configurations.filter(({ threadingMode }) => threadingMode === 'ST');
    if (index === 2) Object.assign(run.configurations[0].results[0], { status: 'failed', durationSeconds: null, error: 'Fictional branch failure' });
    for (const config of run.configurations) for (const result of config.results) {
      if (result.status === 'completed') result.durationSeconds = Number((result.durationSeconds * (0.9 + index / 100)).toFixed(2));
    }
    return run;
  });
  const runs = [...canonical, ...branches];
  return {
    metadata: { schemaVersion: 2, repository: 'https://github.com/ROCm/rocm-systems', isBeta: true, canonicalBranch: 'develop' },
    index: { generatedAt: '2026-10-05T12:00:00.000Z', runFiles: runs.map(({ id }) => `runs/${id}.json`) },
    catalogs, runs,
  };
}

export async function writeSchema2FixtureDirectory(directory) {
  const publication = createSchema2Publication();
  await mkdir(directory, { recursive: true });
  // This function deliberately targets test data only; never point it at published data.
  for (const entry of await readdir(directory)) await rm(path.join(directory, entry), { recursive: true, force: true });
  const files = { 'metadata.json': publication.metadata, 'index.json': publication.index, ...publication.catalogs,
    ...Object.fromEntries(publication.runs.map((run) => [`runs/${run.id}.json`, run])), };
  for (const [name, value] of Object.entries(files)) {
    await mkdir(path.dirname(path.join(directory, name)), { recursive: true });
    await writeFile(path.join(directory, name), `${JSON.stringify(value, null, 2)}\n`);
  }
  return publication;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await writeSchema2FixtureDirectory(fileURLToPath(new URL('./data/', import.meta.url)));
  console.log('Wrote fictional schema-2 test fixtures (24 develop, 20 branches).');
}
