const CATALOG_PATH = 'test-catalogs/synthetic-core-v2.json';
const KEYS = ['gfx1250:ST', 'gfx950:MT'];
const catalog = {
  id: 'synthetic-core-v2',
  tests: [
    { id: 'gemm-f16-1024', suite: 'Triton', name: 'Fictional GEMM FP16', problem: { operation: 'GEMM', m: 1024 } },
    { id: 'gemm-bf16-4096', suite: 'Triton', name: 'Fictional GEMM BF16', problem: { operation: 'GEMM', m: 4096 } },
  ],
  configurations: Object.fromEntries(KEYS.map((key) => [key, ['gemm-f16-1024', 'gemm-bf16-4096']])),
};
function syntheticRun(index) {
  const day = new Date(Date.UTC(2026, 0, 1) + index * 86400_000).toISOString();
  const id = `synthetic-run-${String(index).padStart(4, '0')}`;
  return { id, comparisonId: id, testCatalog: CATALOG_PATH, plugin: { id: 'vanilla', name: 'Vanilla' },
    source: { branch: 'develop', commit: index.toString(16).padStart(40, '0'), committedAt: day, message: `Fictional queue test ${index}` },
    execution: { completedAt: day, trigger: 'auto', machine: 'fictional-queue-node' },
    environment: [{ key: 'sdk', label: 'Fictional SDK', value: 'test-only' }],
    configurations: KEYS.map((key) => {
      const [target, mode] = key.split(':');
      return { target, mode, results: catalog.configurations[key].map((testId, testIndex) => ({ testId, status: 'completed', durationSeconds: 1 + testIndex + index / 1000, error: null })) };
    }),
  };
}
// Request-queue stress data, generated in memory only; not shipped by Vite.
export function createSyntheticDataset(runCount = 500) {
  const runFiles = Array.from({ length: runCount }, (_, index) => `runs/synthetic-run-${String(index).padStart(4, '0')}.json`);
  const bodies = new Map([
    ['https://dashboard.test/data/metadata.json', { schemaVersion: 2, repository: 'https://example.test/repo', isBeta: false, canonicalBranch: 'develop' }],
    ['https://dashboard.test/data/index.json', { generatedAt: '2027-06-01T00:00:00.000Z', runFiles }],
    [`https://dashboard.test/data/${CATALOG_PATH}`, catalog],
  ]);
  runFiles.forEach((runFile, index) => bodies.set(`https://dashboard.test/data/${runFile}`, syntheticRun(index)));
  return { runCount, runFiles, catalogPath: CATALOG_PATH, metadataUrl: 'https://dashboard.test/data/metadata.json', indexUrl: 'https://dashboard.test/data/index.json', bodies };
}
