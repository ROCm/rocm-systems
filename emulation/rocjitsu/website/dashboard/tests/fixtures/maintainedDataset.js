import { fixtureCatalog, fixtureProvenance, fixtureSummary } from './inputFormat.js';
import { fixtureRunPath } from './runPath.js';
// Fictional deterministic maintained-schema measurements. Test/fixtures mode ONLY.
import { mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const CONFIGURATION_KEYS = ['gfx1250/single', 'gfx1250/default', 'gfx950/single', 'gfx950/default'];
const definitions = [
  ['a', 'Triton', 'Fictional GEMM', { operation: 'GEMM', m: 1024 }],
  ['b', 'Llama', 'Fictional decode', { tokens: 128 }],
  ['c', 'Triton', 'Fictional retired workload', { size: 64 }],
  ['d', 'Triton', 'Fictional added workload D', { size: 256 }],
  ['e', 'Llama', 'Fictional added workload E', { size: 512 }],
].map(([id, suite, name, problem]) => ({ id, suite, name, operation: problem.operation ?? 'Fictional computation', problem }));
const sha = (index) => `${index.toString(16).padStart(4, '0')}f1c7`.repeat(5);
const iso = (day, hour = 6, minute = 0) => new Date(Date.UTC(2026, 8, 11 + day, hour, minute)).toISOString();
const catalogFor = (current) => fixtureCatalog(current ? 'fictional-current' : 'fictional-old',
  structuredClone(definitions.filter(({ id }) => (current ? ['a', 'b', 'd', 'e'] : ['a', 'b', 'c']).includes(id))),
  Object.fromEntries(CONFIGURATION_KEYS.map((key) => [key,
    current ? (key === 'gfx950/default' ? ['a', 'b', 'd'] : ['a', 'b', 'd', 'e']) : ['a', 'b', 'c'],
  ])));


export function createMaintainedPublication() {
  const catalogs = Object.fromEntries([false, true].map((current) => {
    const catalog = catalogFor(current);
    return [catalog.benchmark_suite, catalog];
  }));
  const canonical = Array.from({ length: 24 }, (_, index) => {
    const current = index >= 8;
    const testCatalog = `fictional-${current ? 'current' : 'old'}`;
    const id = `fictional-develop-${String(index).padStart(2, '0')}`;
    return {
      schema_version: 1, publication: { id, workflow_commit_sha: 'a'.repeat(40), tensile_commit_sha: 'b'.repeat(40),
      source: { branch: 'develop', message: `FICTIONAL develop measurement ${index}` },
      trigger: index === 21 ? 'manual' : 'auto', machine_label: index < 16 ? 'fictional-node-a' : 'fictional-node-b' },
      execution_summary: fixtureSummary(catalogs[testCatalog].benchmark_suite, iso(index, 5), index === 21 ? '2026-10-05T10:00:00.000Z' : iso(index)),
      provenance: fixtureProvenance(sha(index + 1), iso(index, 4), index < 16 ? 'test-a' : 'test-b'),
      benchmark_results: CONFIGURATION_KEYS.filter((key) => !(index === 12 && key === 'gfx950/default')).flatMap((key, configIndex) => {
        const [target, policy] = key.split('/');
        return catalogs[testCatalog].membership[target][policy].map((testId, testIndex) => {
            const status = (index === 8 && policy === 'single' && testId === 'd') || (index === 9 && policy === 'single' && testId === 'e') ? 'failed'
              : index === 8 && testId === 'e' ? 'timeout' : 'completed';
            const durationSeconds = status !== 'completed' ? null : Number((20 + testIndex * 10 + configIndex * 5 - index * 0.2).toFixed(2));
            return { ...catalogs[testCatalog].benchmark_definitions[testId], id: testId, target, thread_policy: policy, exit_code: status === 'completed' ? 0 : 1, status, timing_results_s: status === 'completed' ? [durationSeconds] : [], error: status === 'completed' ? null : 'Fictional diagnostic' };
          });
      }),
    };
  });
  const branches = Array.from({ length: 20 }, (_, index) => {
    const run = structuredClone(canonical[23]);
    run.publication.id = `fictional-branch-${String(index + 1).padStart(2, '0')}`;
    run.publication.source = { branch: `fictional/optimization-${String(index + 1).padStart(2, '0')}`, message: `FICTIONAL branch ${index + 1}`,
      ...(index % 2 === 0 ? { base: { branch: 'develop', commit: canonical[20].provenance.rocjitsu.rocjitsu_commit_sha }, pull_request: { number: 1000 + index } } : {}),
    };
    run.execution_summary.finished_at = `2026-10-05T11:${String(index).padStart(2, '0')}:00.000Z`;
    run.publication.trigger = 'manual';
    run.publication.machine_label = 'fictional-branch-node';
    run.provenance = fixtureProvenance(sha(100 + index), '2026-10-05T09:00:00.000Z', 'test-branch');
    run.execution_summary.timestamp = '2026-10-05T10:00:00.000Z';
    if (index === 1) run.benchmark_results = run.benchmark_results.filter(({ thread_policy }) => thread_policy === 'single');
    if (index === 2) Object.assign(run.benchmark_results[0], { status: 'failed', timing_results_s: [], error: 'Fictional branch failure' });
    for (const result of run.benchmark_results) {
      if (result.status === 'completed') {
        result.timing_results_s = [Number((result.timing_results_s[0] * (0.9 + index / 100)).toFixed(2))];
      }
    }
    return run;
  });
  const runs = [...canonical, ...branches];
  return {
    index: { generatedAt: '2026-10-05T12:00:00.000Z', runFiles: runs.map(fixtureRunPath) },
    runs,
  };
}

export async function writeMaintainedFixtureDirectory(directory) {
  return writeFixturePublication(directory, createMaintainedPublication());
}

export async function writeFixturePublication(directory, publication) {
  await mkdir(directory, { recursive: true });
  // This function deliberately targets test data only; never point it at published data.
  for (const entry of await readdir(directory)) await rm(path.join(directory, entry), { recursive: true, force: true });
  const files = { 'index.json': publication.index,
    ...Object.fromEntries(publication.runs.map((run, index) => [publication.index.runFiles[index], run])), };
  for (const [name, value] of Object.entries(files)) {
    await mkdir(path.dirname(path.join(directory, name)), { recursive: true });
    await writeFile(path.join(directory, name), `${JSON.stringify(value, null, 2)}\n`);
  }
  return publication;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await writeMaintainedFixtureDirectory(fileURLToPath(new URL('./data/', import.meta.url)));
  console.log('Wrote fictional maintained-schema test fixtures (24 develop, 20 branches).');
}
