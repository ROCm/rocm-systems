import { cp, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from 'vitest';
import { validateDashboardDataDirectory } from '../../scripts/validate-dashboard-data.mjs';

const fixtureDataDirectory = fileURLToPath(new URL('../fixtures/data/', import.meta.url));

test('reports publication policy differences without relying on website filtering', async () => {
  const error = await validateDashboardDataDirectory(fixtureDataDirectory)
    .catch((validationError) => validationError);

  expect(error.message).toContain(
    'benchmark-202608270530-0db03af1 uses machine sjc-rocjitsu-perf-02',
  );
  expect(error.message).toContain(
    'benchmark-202608290530-19872076 uses a different environment',
  );
});

test('rejects generated data that the website would skip', async () => {
  const temporaryDirectory = await mkdtemp(path.join(tmpdir(), 'rocjitsu-dashboard-data-'));
  try {
    await cp(fixtureDataDirectory, temporaryDirectory, { recursive: true });
    const index = JSON.parse(await readFile(path.join(temporaryDirectory, 'index.json'), 'utf8'));
    const runPath = path.join(temporaryDirectory, index.runFiles[0]);
    const run = JSON.parse(await readFile(runPath, 'utf8'));
    const completedResult = run.targets
      .flatMap(({ results }) => results)
      .find(({ status }) => status === 'completed');
    completedResult.error = 'Unexpected diagnostic';
    await writeFile(runPath, `${JSON.stringify(run, null, 2)}\n`);

    await expect(validateDashboardDataDirectory(temporaryDirectory)).rejects.toThrow(
      `Completed result ${completedResult.testId} cannot contain an error`,
    );
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
});

