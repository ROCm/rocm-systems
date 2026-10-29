#!/usr/bin/env node
// Local-only consumer processing: never publishes or fabricates benchmark data.
import { lstat, readFile, mkdir, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { adaptPublishedDashboardData, validatePublishedManifest, RUN_FILE_PATTERN } from '../src/data/dashboardDataAdapter.js';
import { parsePublishedJson } from '../src/data/publishedJson.js';

async function rejectSymbolicLinks(file, root) {
  let current = path.resolve(file);
  while (true) {
    if ((await lstat(current)).isSymbolicLink()) {
      throw new Error(`Resource path contains a symbolic link: ${current}`);
    }
    if (current === root) return;
    current = path.dirname(current);
  }
}

async function readJson(file, label, root) {
  try {
    await rejectSymbolicLinks(file, root);
    const resolved = await realpath(file);
    const relative = path.relative(root, resolved);
    if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
      throw new Error('Resource resolves outside the data directory');
    }
    return parsePublishedJson(await readFile(resolved, 'utf8'));
  } catch (error) {
    throw new Error(`Unable to read ${label} ${file}: ${error.message}`, { cause: error });
  }
}

export async function loadDashboardDataDirectory(directory) {
  const requestedRoot = path.resolve(directory);
  if ((await lstat(requestedRoot)).isSymbolicLink()) {
    throw new Error(`Dashboard data directory is a symbolic link: ${requestedRoot}`);
  }
  const root = await realpath(requestedRoot);
  const index = await readJson(path.join(root, 'index.json'), 'dashboard data index', root);

  validatePublishedManifest(index);

  const runResults = await Promise.all(index.runFiles.map(async (runFile) => {
    if (typeof runFile !== 'string' || !RUN_FILE_PATTERN.test(runFile)) {
      return { run: null, error: new Error(`Invalid run filename: ${String(runFile)}`) };
    }
    try {
      return {
        run: await readJson(path.join(root, runFile), 'run file', root),
        error: null,
      };
    } catch (error) {
      return { run: null, error };
    }
  }));

  const result = adaptPublishedDashboardData({
    index,
    runs: runResults.map(({ run }) => run),
    runErrors: runResults.map(({ error }) => error),
  });

  return result;
}



function isInside(directory, destination) {
  const relative = path.relative(directory, destination);
  return !relative || !relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative);
}

// Resolve existing ancestors before any mkdir, including missing output parents.
async function resolveDestination(destination) {
  let ancestor = destination;
  for (;;) {
    try { return path.resolve(await realpath(ancestor), path.relative(ancestor, destination)); }
    catch (error) {
      const parent = path.dirname(ancestor);
      if (error.code !== 'ENOENT' || parent === ancestor) throw error;
      ancestor = parent;
    }
  }
}

export async function processDashboardDataDirectory(directory, { output } = {}) {
  const inputPath = path.resolve(directory);
  const root = await realpath(inputPath);
  const outputPath = output ? path.resolve(output) : null;
  if (outputPath && (isInside(inputPath, outputPath) || isInside(root, await resolveDestination(outputPath)))) {
    throw new Error('Processing output must be outside the input data directory');
  }
  const { data, sourceData } = await loadDashboardDataDirectory(inputPath);
  // JSON has no Set type, so serialize backfillRunIds as an array.
  const result = { data: { ...data, backfillRunIds: [...data.backfillRunIds] }, sourceData };
  if (outputPath) {
    await mkdir(path.dirname(outputPath), { recursive: true });
    await writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`, { flag: 'wx' });
  }
  return result;
}

async function main() {
  const [directory, option, output, ...extra] = process.argv.slice(2);
  if (!directory || extra.length || option && (option !== '--output' || !output)) {
    throw new Error('Usage: node scripts/process-dashboard-data.mjs <data-directory> [--output <new-file-outside-input>]');
  }
  const result = await processDashboardDataDirectory(directory, { output });
  if (output) console.log(`Processed maintained schema 1: ${result.data.runs.length} develop, ${result.data.allRuns.length} total attempts. Output: ${path.resolve(output)}`);
  else console.log(JSON.stringify(result, null, 2));
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
