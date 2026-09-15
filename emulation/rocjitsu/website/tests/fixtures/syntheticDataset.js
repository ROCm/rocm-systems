const CATALOG_PATH = 'test-catalogs/synthetic-core-v1.json';
const TARGETS = ['gfx1250', 'gfx950'];

const catalog = {
  id: 'synthetic-core-v1',
  tests: [
    { id: 'gemm-f16-1024', suite: 'Triton', name: 'GEMM FP16 1024³', problem: { operation: 'GEMM', dataType: 'fp16', m: 1024 } },
    { id: 'gemm-bf16-4096', suite: 'Triton', name: 'GEMM BF16 4096³', problem: { operation: 'GEMM', dataType: 'bf16', m: 4096 } },
  ],
  targets: Object.fromEntries(TARGETS.map((target) => [target, ['gemm-f16-1024', 'gemm-bf16-4096']])),
};

function syntheticRun(index) {
  const day = new Date(Date.UTC(2026, 0, 1) + index * 86_400_000).toISOString();
  const id = `synthetic-run-${String(index).padStart(4, '0')}`;
  return {
    id,
    comparisonId: id,
    testCatalog: CATALOG_PATH,
    plugin: { id: 'vanilla', name: 'Vanilla' },
    source: {
      branch: 'develop',
      commit: `${String(index).padStart(8, '0')}${'0'.repeat(32)}`,
      committedAt: day,
      message: `Synthetic commit ${index}`,
    },
    execution: { completedAt: day, trigger: 'auto', machine: 'synthetic-node' },
    environment: [{ key: 'rocm', label: 'ROCm SDK', value: '7.2.0' }],
    targets: TARGETS.map((target) => ({
      id: target,
      results: catalog.targets[target].map((testId, testIndex) => ({
        testId,
        status: 'completed',
        durationSeconds: Number((1 + testIndex + index / 1000).toFixed(3)),
      })),
    })),
  };
}

// A dataset large enough to exercise the request queue rather than the parser: 500 runs is one
// request per run plus metadata, index, and the shared catalog.
export function createSyntheticDataset(runCount = 500) {
  const runFiles = Array.from({ length: runCount }, (_, index) => `runs/synthetic-run-${String(index).padStart(4, '0')}.json`);
  const bodies = new Map([
    ['https://dashboard.test/data/metadata.json', { schemaVersion: 1, repository: 'https://example.test/repo', isBeta: false }],
    ['https://dashboard.test/data/index.json', { generatedAt: '2027-06-01T00:00:00.000Z', runFiles }],
    [`https://dashboard.test/data/${CATALOG_PATH}`, catalog],
  ]);
  runFiles.forEach((runFile, index) => {
    bodies.set(`https://dashboard.test/data/${runFile}`, syntheticRun(index));
  });

  return {
    runCount,
    runFiles,
    catalogPath: CATALOG_PATH,
    metadataUrl: 'https://dashboard.test/data/metadata.json',
    indexUrl: 'https://dashboard.test/data/index.json',
    bodies,
  };
}
