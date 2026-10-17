import { expect, test } from 'vitest';
import { mkdir, mkdtemp, readFile, readdir, rm, symlink, writeFile, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { writeSchema2FixtureDirectory } from '../fixtures/schema2Dataset.js';
import { validateDashboardDataDirectory } from '../../scripts/validate-dashboard-data.mjs';
const processing = await import('../../scripts/process-dashboard-data.mjs').catch(() => ({}));

test('local processing validates before output and preserves source, modes and serializable backfill identities', async () => {
  expect(typeof processing.processDashboardDataDirectory).toBe('function');
  const root = await mkdtemp(path.join(tmpdir(), 'schema2-processing-'));
  try {
    const source = await writeSchema2FixtureDirectory(path.join(root, 'input'));
    const output = path.join(root, 'processed.json');
    const result = await processing.processDashboardDataDirectory(path.join(root, 'input'), { output });
    expect(result.sourceData).toEqual(source);
    expect(result.data.runs).toHaveLength(24);
    expect(result.data.allRuns).toHaveLength(44);
    expect(result.data.backfillRunIds).toContain('fictional-develop-21');
    expect(JSON.parse(await readFile(output, 'utf8'))).toEqual(result);
    await expect(processing.processDashboardDataDirectory(path.join(root, 'input'), { output: path.join(root, 'input', 'index.json') })).rejects.toThrow(/outside.*input/i);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test.each([
  ['symlinked output parent', false, true, false],
  ['symlinked output ancestor with missing descendants', false, true, true],
  ['symlinked input root', true, false, true],
  ['symlinked input and output roots', true, true, true],
])('DATA-05 rejects input-bound output through %s before creating directories or files', async (_label, inputAlias, outputAlias, nested) => {
  const root = await mkdtemp(path.join(tmpdir(), 'schema2-processing-boundary-'));
  try {
    const input = path.join(root, 'input');
    await writeSchema2FixtureDirectory(input);
    const before = (await readdir(input, { recursive: true })).sort();
    const directory = inputAlias ? path.join(root, 'input-alias') : input;
    if (inputAlias) await symlink(input, directory, 'dir');
    const destination = outputAlias ? path.join(root, 'outside-alias') : input;
    if (outputAlias) await symlink(input, destination, 'dir');
    const output = nested ? path.join(destination, 'new', 'nested', 'processed.json') : path.join(destination, 'processed.json');
    await expect(processing.processDashboardDataDirectory(directory, { output }).then(() => 'processing succeeded')).rejects.toThrow(/outside.*input/i);
    await expect(access(output)).rejects.toMatchObject({ code: 'ENOENT' });
    expect((await readdir(input, { recursive: true })).sort()).toEqual(before);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('DATA-05 permits genuinely external symlinked output directories without overwriting files', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'schema2-processing-external-'));
  try {
    const input = path.join(root, 'input');
    await writeSchema2FixtureDirectory(input);
    const outside = path.join(root, 'outside');
    await mkdir(outside);
    const alias = path.join(root, 'outside-alias');
    await symlink(outside, alias, 'dir');
    const output = path.join(alias, 'new', 'nested', 'processed.json');
    const result = await processing.processDashboardDataDirectory(input, { output });
    const original = await readFile(output, 'utf8');
    expect(JSON.parse(original)).toEqual(result);
    await expect(processing.processDashboardDataDirectory(input, { output })).rejects.toMatchObject({ code: 'EEXIST' });
    expect(await readFile(output, 'utf8')).toBe(original);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('local validation rejects symlink escapes and invalid snapshots never create processing output', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'schema2-processing-security-'));
  try {
    const directory = path.join(root, 'input');
    const source = await writeSchema2FixtureDirectory(directory);
    const runPath = path.join(directory, source.index.runFiles[0]);
    const outside = path.join(root, 'outside.json');
    await writeFile(outside, JSON.stringify(source.runs[0]));
    await rm(runPath); await symlink(outside, runPath);
    await expect(validateDashboardDataDirectory(directory)).rejects.toThrow(/symbolic link/i);
    const output = path.join(root, 'must-not-exist.json');
    await expect(processing.processDashboardDataDirectory(directory, { output })).rejects.toThrow();
    await expect(access(output)).rejects.toThrow();
  } finally { await rm(root, { recursive: true, force: true }); }
});
