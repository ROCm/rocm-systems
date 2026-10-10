import {
  cp, mkdtemp, readFile, rename, rm, symlink, writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from 'vitest';
import { validateDashboardDataDirectory } from '../../scripts/validate-dashboard-data.mjs';

const fixtureDataDirectory = fileURLToPath(new URL('../fixtures/data/', import.meta.url));

test('accepts historical environment changes in an otherwise valid publication', async () => {
  const result = await validateDashboardDataDirectory(fixtureDataDirectory);

  const index = JSON.parse(await readFile(path.join(fixtureDataDirectory, 'index.json'), 'utf8'));
  expect(result.sourceData.runs).toHaveLength(index.runFiles.length);
});

test('accepts an empty publication without run or catalog directories', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'dashboard-empty-'));
  try {
    const index = JSON.parse(await readFile(path.join(fixtureDataDirectory, 'index.json'), 'utf8'));
    await writeFile(path.join(directory, 'index.json'), JSON.stringify({ ...index, runFiles: [] }));
    const result = await validateDashboardDataDirectory(directory);
    expect(result.sourceData.runs).toEqual([]);
    expect(result.data.runs).toEqual([]);
    await writeFile(path.join(directory, 'metadata.json'), 'not valid JSON');
    expect(await validateDashboardDataDirectory(directory)).toEqual(result);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('rejects generated data that the website also rejects', async () => {
  const temporaryDirectory = await mkdtemp(path.join(tmpdir(), 'rocjitsu-dashboard-data-'));
  try {
    await cp(fixtureDataDirectory, temporaryDirectory, { recursive: true });
    const index = JSON.parse(await readFile(path.join(temporaryDirectory, 'index.json'), 'utf8'));
    const runPath = path.join(temporaryDirectory, index.runFiles[0]);
    const run = JSON.parse(await readFile(runPath, 'utf8'));
    const completedResult = run.configurations
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

test.each(['run file', 'run directory', 'catalog file'])(
  'rejects a symbolic-link %s inside the publication',
  async (resource) => {
    const temporaryDirectory = await mkdtemp(path.join(tmpdir(), 'rocjitsu-dashboard-data-'));
    try {
      await cp(fixtureDataDirectory, temporaryDirectory, { recursive: true });
      const index = JSON.parse(await readFile(path.join(temporaryDirectory, 'index.json'), 'utf8'));
      const runPath = path.join(temporaryDirectory, index.runFiles[0]);
      const run = JSON.parse(await readFile(runPath, 'utf8'));
      const selected = resource === 'run file'
        ? runPath
        : resource === 'run directory'
          ? path.dirname(runPath)
          : path.join(temporaryDirectory, run.testCatalog);
      const target = `${selected}-target`;
      await rename(selected, target);
      await symlink(path.basename(target), selected, resource === 'run directory' ? 'dir' : 'file');

      await expect(validateDashboardDataDirectory(temporaryDirectory)).rejects.toThrow(
        /symbolic link/i,
      );
    } finally {
      await rm(temporaryDirectory, { recursive: true, force: true });
    }
  },
);
