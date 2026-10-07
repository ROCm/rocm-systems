#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  CATALOG_FILE_PATTERN,
  RUN_FILE_PATTERN,
  validatePublishedDashboardData,
} from '../src/data/dashboardValidation.js';

async function readJson(file, label) {
  try {
    return JSON.parse(await readFile(file, 'utf8'));
  } catch (error) {
    throw new Error(`Unable to read ${label} ${file}: ${error.message}`, { cause: error });
  }
}

export async function validateDashboardDataDirectory(directory) {
  const root = path.resolve(directory);
  const metadata = await readJson(path.join(root, 'metadata.json'), 'dashboard metadata');
  const index = await readJson(path.join(root, 'index.json'), 'dashboard data index');

  if (!index || !Array.isArray(index.runFiles)) {
    return validatePublishedDashboardData({ metadata, index, runs: [] });
  }

  const runResults = await Promise.all(index.runFiles.map(async (runFile) => {
    if (typeof runFile !== 'string' || !RUN_FILE_PATTERN.test(runFile)) {
      return { run: null, error: new Error(`Invalid run filename: ${String(runFile)}`) };
    }
    try {
      return {
        run: await readJson(path.join(root, runFile), 'run file'),
        error: null,
      };
    } catch (error) {
      return { run: null, error };
    }
  }));

  const catalogPaths = [...new Set(runResults
    .map((result) => result.run?.testCatalog)
    .filter((catalogPath) => (
      typeof catalogPath === 'string' && CATALOG_FILE_PATTERN.test(catalogPath)
    )))];
  const catalogResults = await Promise.all(catalogPaths.map(async (catalogPath) => {
    try {
      return {
        catalogPath,
        catalog: await readJson(path.join(root, catalogPath), 'test catalog'),
        error: null,
      };
    } catch (error) {
      return { catalogPath, catalog: null, error };
    }
  }));

  const result = validatePublishedDashboardData({
    metadata,
    index,
    runs: runResults.map(({ run }) => run),
    runErrors: runResults.map(({ error }) => error),
    catalogs: Object.fromEntries(catalogResults.map(({ catalogPath, catalog }) => (
      [catalogPath, catalog]
    ))),
    catalogErrors: Object.fromEntries(catalogResults
      .filter(({ error }) => error)
      .map(({ catalogPath, error }) => [catalogPath, error])),
  });

  return result;
}

async function main() {
  const [directory, ...extraArguments] = process.argv.slice(2);
  if (!directory || extraArguments.length > 0) {
    throw new Error('Usage: node scripts/validate-dashboard-data.mjs <data-directory>');
  }

  const result = await validateDashboardDataDirectory(directory);
  console.log(`Dashboard data is valid (${result.sourceData.runs.length} run files).`);
}

const invokedAsScript = process.argv[1]
  && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;
if (invokedAsScript) {
  main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
