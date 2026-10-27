import { readMaintainedExamplePublication } from '../helpers/maintainedExamplePublication.js';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { expect, test } from 'vitest';
import { loadDashboardDataFiles } from '../../src/data/dashboardData.js';
import { loadDashboardDataDirectory } from '../../scripts/process-dashboard-data.mjs';
import { processDashboardDataDirectory } from '../../scripts/process-dashboard-data.mjs';

function publicationBodies() {
  const { index, runs: [run] } = readMaintainedExamplePublication();
  return { 'index.json': JSON.stringify(index), [index.runFiles[0]]: JSON.stringify(run) };
}
const load = (bodies) => loadDashboardDataFiles({
  indexUrl: 'https://fixture.test/data/index.json',
  retryDelaysMs: [],
  fetch: async (url) => ({ ok: true, text: async () => bodies[new URL(url).pathname.slice('/data/'.length)] }),
});

test.each(['9007199254740993', '9.007199254740993e15', '1e-400'])('all text loaders reject unsupported numeric token %s', async (token) => {
  const bodies = publicationBodies();
  bodies['runs/default-branch/run-001.json'] = bodies['runs/default-branch/run-001.json'].replace('"problem":{', `"problem":{"unsafe":${token},`);
  await expect(load(bodies)).rejects.toThrow(/numeric/i);
  const directory = await mkdtemp(path.join(process.env.TMPDIR ?? os.tmpdir(), 'numeric-json-'));
  try {
    for (const [name, body] of Object.entries(bodies)) {
      await mkdir(path.dirname(path.join(directory, name)), { recursive: true });
      await writeFile(path.join(directory, name), body);
    }
    await expect(loadDashboardDataDirectory(directory)).rejects.toThrow(/numeric/i);
    await expect(processDashboardDataDirectory(directory)).rejects.toThrow(/numeric/i);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('browser text boundary rejects duplicate decoded object keys', async () => {
  const bodies = publicationBodies();
  bodies['runs/default-branch/run-001.json'] = bodies['runs/default-branch/run-001.json'].replace('"schema_version":', '"schema_ve\\u0072sion":1,"schema_version":');
  await expect(load(bodies)).rejects.toThrow(/Duplicate JSON key/);
});

test('CLI and processor reject duplicate JSON declarations before normalization', async () => {
  const directory = await mkdtemp(path.join(process.env.TMPDIR ?? os.tmpdir(), 'maintained-json-'));
  try {
    const bodies = publicationBodies();
    bodies['index.json'] = bodies['index.json'].replace('"runFiles":', '"runFiles":[],"runFiles":');
    for (const [name, body] of Object.entries(bodies)) {
      await mkdir(path.dirname(path.join(directory, name)), { recursive: true });
      await writeFile(path.join(directory, name), body);
    }
    await expect(loadDashboardDataDirectory(directory)).rejects.toThrow(/Duplicate JSON key/);
    await expect(processDashboardDataDirectory(directory)).rejects.toThrow(/Duplicate JSON key/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test.each([undefined, 2, 99, '1'])('browser rejects unsupported or missing version %s', async (version) => {
  const bodies = publicationBodies();
  const runPath = 'runs/default-branch/run-001.json';
  const run = JSON.parse(bodies[runPath]);
  if (version === undefined) delete run.schema_version;
  else run.schema_version = version;
  bodies[runPath] = JSON.stringify(run);
  await expect(load(bodies)).rejects.toThrow(/unsupported schema version/i);
});
