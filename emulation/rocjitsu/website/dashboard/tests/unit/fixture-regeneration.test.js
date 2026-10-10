import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from 'vitest';
import { validateDashboardDataDirectory } from '../../scripts/validate-dashboard-data.mjs';

const checkedIn = fileURLToPath(new URL('../fixtures/data/', import.meta.url));
const generator = fileURLToPath(new URL('../fixtures/regenerate-fixtures.js', import.meta.url));
const jsonFiles = async (directory) => (await readdir(directory, { recursive: true })).filter((name) => name.endsWith('.json')).sort();

test('one regeneration command reproduces the complete checked-in publication', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'dashboard-fixture-regeneration-'));
  try {
    await writeFile(path.join(directory, 'obsolete.json'), '{}\n');
    const result = spawnSync(process.execPath, [generator, directory], { encoding: 'utf8' });
    expect(result.status, result.stderr).toBe(0);
    const files = await jsonFiles(directory);
    expect(files).toEqual(await jsonFiles(checkedIn));
    for (const name of files) {
      expect(await readFile(path.join(directory, name), 'utf8'), name).toBe(await readFile(path.join(checkedIn, name), 'utf8'));
    }
    const validated = await validateDashboardDataDirectory(directory);
    expect(validated.sourceData.runs).toHaveLength(45);
    expect(validated.data.runs).toHaveLength(24);
    expect(validated.data.allRuns).toHaveLength(45);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
