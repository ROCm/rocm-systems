import {
  CATALOG_FILE_PATTERN,
  RUN_FILE_PATTERN,
  validatePublishedDashboardData,
} from './dashboardValidation.js';

export { loadDashboardData, validatePublishedDashboardData } from './dashboardValidation.js';

function hasText(value) {
  return typeof value === 'string' && Boolean(value.trim());
}

// A dataset of several hundred runs is one HTTP request per run. Browsers queue beyond their own
// per-host limit anyway, and an unbounded fan-out makes every request share the same slow ramp, so
// the loader keeps a fixed number of requests in flight and reports progress as they settle.
export const MAX_CONCURRENT_RUN_REQUESTS = 8;
export const DEFAULT_REQUEST_TIMEOUT_MS = 20_000;
export const DEFAULT_LOAD_TIMEOUT_MS = 60_000;

const loadCancellation = Symbol('dashboard-load-cancellation');

class LoadCancelledError extends Error {
  constructor() {
    super('Dashboard data loading was cancelled');
    this.name = 'AbortError';
    this[loadCancellation] = true;
  }
}

export function isLoadCancelled(error) {
  return error?.[loadCancellation] === true;
}

function throwIfCancelled(signal) {
  if (signal?.aborted) throw new LoadCancelledError();
}

async function fetchJsonResource(url, {
  fetchImpl,
  signal,
  timeoutMs,
  cache,
  resourceType,
}) {
  throwIfCancelled(signal);
  const controller = new AbortController();
  const forwardAbort = () => controller.abort();
  signal?.addEventListener('abort', forwardAbort, { once: true });
  let timedOut = false;
  const timer = Number.isFinite(timeoutMs) && timeoutMs > 0
    ? setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs)
    : null;

  try {
    const response = await fetchImpl(String(url), {
      signal: controller.signal,
      ...(cache ? { cache } : {}),
    });
    if (!response.ok) {
      throw new Error(`Unable to load ${resourceType} ${url} (${response.status} ${response.statusText})`);
    }
    const contentType = response.headers?.get?.('content-type');
    if (contentType && !contentType.toLowerCase().includes('json')) {
      if (resourceType === 'dashboard metadata' || resourceType === 'dashboard data index') {
        throw new Error('No available test data');
      }
      throw new Error(`Unable to parse ${resourceType} ${url} as JSON: received ${contentType}`);
    }
    try {
      return await response.json();
    } catch (parseError) {
      throw new Error(`Unable to parse ${resourceType} ${url} as JSON: ${parseError.message}`, { cause: parseError });
    }
  } catch (error) {
    if (signal?.aborted) throw new LoadCancelledError();
    if (timedOut) throw new Error(`Timed out after ${timeoutMs} ms loading ${resourceType} ${url}`, { cause: error });
    throw error;
  } finally {
    if (timer) clearTimeout(timer);
    signal?.removeEventListener('abort', forwardAbort);
  }
}

async function mapWithConcurrency(items, limit, worker) {
  const results = new Array(items.length);
  let nextIndex = 0;
  const workerCount = Math.max(1, Math.min(limit, items.length));
  await Promise.all(Array.from({ length: workerCount }, async () => {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await worker(items[index], index);
    }
  }));
  return results;
}

export async function loadDashboardDataFiles({
  metadataUrl,
  indexUrl,
  onManifest,
  onProgress,
  signal,
  fetch: fetchImpl = globalThis.fetch.bind(globalThis),
  concurrency = MAX_CONCURRENT_RUN_REQUESTS,
  requestTimeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
  loadTimeoutMs = DEFAULT_LOAD_TIMEOUT_MS,
}) {
  const loadController = new AbortController();
  const forwardAbort = () => loadController.abort();
  if (signal?.aborted) forwardAbort();
  else signal?.addEventListener('abort', forwardAbort, { once: true });
  let loadTimedOut = false;
  const loadTimer = Number.isFinite(loadTimeoutMs) && loadTimeoutMs > 0
    ? setTimeout(() => {
      loadTimedOut = true;
      loadController.abort();
    }, loadTimeoutMs)
    : null;

  try {
  // metadata.json and index.json are the only mutable documents in the contract, so they must
  // revalidate on every load while immutable runs and catalogs use ordinary HTTP caching.
  const loadSignal = loadController.signal;
  const mutableRequest = { fetchImpl, signal: loadSignal, timeoutMs: requestTimeoutMs, cache: 'no-store' };
  const immutableRequest = { fetchImpl, signal: loadSignal, timeoutMs: requestTimeoutMs };
  const [metadata, index] = await Promise.all([
    fetchJsonResource(metadataUrl, { ...mutableRequest, resourceType: 'dashboard metadata' }),
    fetchJsonResource(indexUrl, { ...mutableRequest, resourceType: 'dashboard data index' }),
  ]);
  if (!index || !Array.isArray(index.runFiles)) {
    throw new Error('Expected the dashboard data index to contain a runFiles array');
  }
  onManifest?.({ ...metadata, generatedAt: index.generatedAt });

  const total = index.runFiles.length;
  let loaded = 0;
  onProgress?.({ loaded, total });

  const runResults = await mapWithConcurrency(index.runFiles, concurrency, async (runFile) => {
    const settle = (result) => {
      loaded += 1;
      onProgress?.({ loaded, total });
      return result;
    };
    if (typeof runFile !== 'string' || !RUN_FILE_PATTERN.test(runFile)) {
      return settle({ run: null, error: new Error(`Invalid run filename: ${String(runFile)}`) });
    }
    try {
      const run = await fetchJsonResource(new URL(runFile, indexUrl), {
        ...immutableRequest,
        resourceType: 'run file',
      });
      return settle({ run, error: null });
    } catch (error) {
      // Cancellation is a caller decision about the whole load, never one skippable run.
      if (isLoadCancelled(error)) throw error;
      return settle({ run: null, error });
    }
  });

  const catalogPaths = [...new Set(runResults
    .map((result) => result.run?.testCatalog)
    .filter((catalogPath) => hasText(catalogPath) && CATALOG_FILE_PATTERN.test(catalogPath)))];
  const catalogResults = await mapWithConcurrency(catalogPaths, concurrency, async (catalogPath) => {
    try {
      const catalog = await fetchJsonResource(new URL(catalogPath, indexUrl), {
        ...immutableRequest,
        resourceType: 'test catalog',
      });
      return { catalogPath, catalog, error: null };
    } catch (error) {
      if (isLoadCancelled(error)) throw error;
      return { catalogPath, catalog: null, error };
    }
  });

  throwIfCancelled(loadSignal);

  return validatePublishedDashboardData({
    metadata,
    index,
    runs: runResults.map((result) => result.run),
    runErrors: runResults.map((result) => result.error),
    catalogs: Object.fromEntries(catalogResults.map((result) => [result.catalogPath, result.catalog])),
    catalogErrors: Object.fromEntries(catalogResults
      .filter((result) => result.error)
      .map((result) => [result.catalogPath, result.error])),
  });
  } catch (error) {
    if (signal?.aborted) throw new LoadCancelledError();
    if (loadTimedOut && isLoadCancelled(error)) {
      throw new Error(`Timed out after ${loadTimeoutMs} ms loading dashboard data`, { cause: error });
    }
    throw error;
  } finally {
    if (loadTimer) clearTimeout(loadTimer);
    signal?.removeEventListener('abort', forwardAbort);
  }
}
