#!/usr/bin/env node
// Local-only consumer processing: never publishes or fabricates benchmark data.
import { mkdir, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { validateDashboardDataDirectory } from './validate-dashboard-data.mjs';

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
  const { data, sourceData } = await validateDashboardDataDirectory(root);
  // JSON has no Set type. Re-loading rebuilds backfillRunIds from canonical runs.
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
  if (output) console.log(`Processed schema 2: ${result.data.runs.length} develop, ${result.data.allRuns.length} Vanilla, ${result.data.pluginRuns.length} total attempts. Output: ${path.resolve(output)}`);
  else console.log(JSON.stringify(result, null, 2));
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
