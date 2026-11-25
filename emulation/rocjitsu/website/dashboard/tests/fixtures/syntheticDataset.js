import { fixtureCatalog, fixtureProvenance, fixtureSummary } from './inputFormat.js';
const KEYS = ['gfx1250/single', 'gfx950/default'];
const catalog = fixtureCatalog('synthetic-core-v2', [
    { id: 'gemm-f16-1024', suite: 'Triton', name: 'Fictional GEMM FP16', operation: 'GEMM', problem: { operation: 'GEMM', m: 1024 } },
    { id: 'gemm-bf16-4096', suite: 'Triton', name: 'Fictional GEMM BF16', operation: 'GEMM', problem: { operation: 'GEMM', m: 4096 } },
  ], Object.fromEntries(KEYS.map((key) => [key, ['gemm-f16-1024', 'gemm-bf16-4096']])));
function syntheticRun(index) {
  const day = new Date(Date.UTC(2026, 0, 1) + index * 86400_000).toISOString();
  const id = `synthetic-run-${String(index).padStart(4, '0')}`;
  return { schema_version: 1, publication: { id, workflow_commit_sha: null, tensile_commit_sha: null,
    source: { branch: 'develop', message: `Fictional queue test ${index}` },
    trigger: 'auto', machine_label: 'fictional-queue-node' },
    execution_summary: fixtureSummary(catalog.benchmark_suite, day),
    provenance: fixtureProvenance(index.toString(16).padStart(40, '0'), day),
    benchmark_results: KEYS.flatMap((key) => {
      const [target, mode] = key.split('/');
      return catalog.membership[target][mode].map((testId, testIndex) => ({ ...catalog.benchmark_definitions[testId], id: testId, target, thread_policy: mode, exit_code: 0, status: 'completed', timing_results_s: [1 + testIndex + index / 1000], error: null }));
    }),
  };
}
// Request-queue stress data, generated in memory only; not shipped by Vite.
export function createSyntheticDataset(runCount = 500) {
  const runFiles = Array.from({ length: runCount }, (_, index) => `runs/default-branch/synthetic-run-${String(index).padStart(4, '0')}.json`);
  const bodies = new Map([
    ['https://dashboard.test/data/index.json', { generatedAt: '2027-06-01T00:00:00.000Z', runFiles }],
  ]);
  runFiles.forEach((runFile, index) => bodies.set(`https://dashboard.test/data/${runFile}`, syntheticRun(index)));
  return { runCount, runFiles, indexUrl: 'https://dashboard.test/data/index.json', bodies };
}
