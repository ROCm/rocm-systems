import { cp, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from 'vitest';
import { validateDashboardDataDirectory } from '../../scripts/validate-dashboard-data.mjs';

const fixtureDataDirectory = fileURLToPath(new URL('../fixtures/data/', import.meta.url));

test('accepts historical environment changes in an otherwise valid publication', async () => {
  const result = await validateDashboardDataDirectory(fixtureDataDirectory);

  expect(result.sourceData.runs).toHaveLength(83);
});

test('rejects generated data that the website also rejects', async () => {
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
