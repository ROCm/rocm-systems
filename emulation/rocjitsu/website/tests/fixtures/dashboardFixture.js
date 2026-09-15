import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const dataDirectory = new URL('./data/', import.meta.url);
const generatedSite = new URL('../../.test-data/', import.meta.url);
const readJson = (name) => JSON.parse(readFileSync(new URL(name, dataDirectory), 'utf8'));

// Shared by unit tests and Vite's fixture mode. Real regression cases stay explicit;
// otherwise unremarkable days are generated to exercise history/selector boundaries.
export function createDashboardFixture() {
  const metadata = readJson('metadata.json');
  const seedIndex = readJson('index.json');
  const cases = seedIndex.runFiles.map(readJson);
  const template = cases.find((run) => run.id === 'benchmark-202607250530-8e0c5183');
  const augustEnvironment = cases.find((run) => run.id === 'benchmark-202608020530-68c7dece').environment;
  const latestDaily = cases.find((run) => run.id === 'benchmark-202608310530-255eabe3');
  const occupiedDays = new Set(cases.map((run) => run.source.committedAt.slice(0, 10)));
  const generated = [];
  const start = Date.UTC(2026, 6, 17, 5, 30);
  const end = Date.UTC(2026, 7, 25, 5, 30);

  for (let time = start; time <= end; time += 86_400_000) {
    const timestamp = new Date(time).toISOString();
    const day = timestamp.slice(0, 10);
    if (occupiedDays.has(day)) continue;
    const run = structuredClone(template);
    run.id = `generated-history-${day}`;
    run.comparisonId = run.id;
    run.source = {
      branch: 'develop',
      commit: createHash('sha1').update(run.id).digest('hex'),
      committedAt: timestamp,
      message: `Generated history for ${day}`,
    };
    run.execution.completedAt = timestamp;
    // The newer comparison environment starts on August 2, not the month boundary.
    if (day >= '2026-08-02') run.environment = structuredClone(augustEnvironment);
    // Interpolate toward a weekly baseline 0.5% faster than the latest daily run.
    // This preserves the range-change scenario without 35 independent recordings.
    const progress = (time - start) / (end - start);
    for (const target of run.targets) {
      const latestResults = latestDaily.targets.find((item) => item.id === target.id).results;
      for (const result of target.results) {
        const endDuration = latestResults.find((item) => item.testId === result.testId).durationSeconds / 1.005;
        result.durationSeconds = Number((result.durationSeconds * (1 - progress) + endDuration * progress).toFixed(3));
      }
    }
    generated.push(run);
  }

  const runs = [...cases, ...generated].sort((a, b) =>
    a.execution.completedAt.localeCompare(b.execution.completedAt) || a.id.localeCompare(b.id));
  const index = { ...seedIndex, runFiles: runs.map((run) => `runs/${run.id}.json`) };
  const catalogs = Object.fromEntries([...new Set(runs.map((run) => run.testCatalog))]
    .map((name) => [name, readJson(name)]));
  return { metadata, index, runs, catalogs };
}

// This path is owned by the fixture generator and ignored by Git. Recreate it so
// removing a scenario cannot leave stale JSON in subsequent dev/test builds.
export function writeDashboardFixtureSite() {
  const { metadata, index, runs, catalogs } = createDashboardFixture();
  const files = {
    'metadata.json': metadata,
    'index.json': index,
    ...catalogs,
    ...Object.fromEntries(runs.map((run) => [`runs/${run.id}.json`, run])),
  };
  rmSync(generatedSite, { recursive: true, force: true });
  for (const [name, value] of Object.entries(files)) {
    const path = fileURLToPath(new URL(`data/${name}`, generatedSite));
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${JSON.stringify(value)}\n`);
  }
  return fileURLToPath(generatedSite);
}
