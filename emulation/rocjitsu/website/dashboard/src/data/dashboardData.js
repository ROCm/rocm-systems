import {
  CATALOG_FILE_PATTERN,
  RUN_FILE_PATTERN,
  validatePublishedDashboardData,
} from './dashboardValidation.js';

export { loadDashboardData, validatePublishedDashboardData } from './dashboardValidation.js';

function hasText(value) {
  return typeof value === 'string' && Boolean(value.trim());
}

function withReloadToken(url, reloadToken) {
  if (!reloadToken) return url;
  const reloadedUrl = new URL(url);
  reloadedUrl.searchParams.set('reload', reloadToken);
  return reloadedUrl;
}

// A dataset of several hundred runs is one HTTP request per run. Browsers queue beyond their own
// per-host limit anyway, and an unbounded fan-out makes every request share the same slow ramp, so
// the loader keeps a fixed number of requests in flight and reports progress as they settle.
export const MAX_CONCURRENT_RUN_REQUESTS = 8;
export const DEFAULT_REQUEST_TIMEOUT_MS = 20_000;
export const DEFAULT_LOAD_TIMEOUT_MS = 60_000;
export const DEFAULT_RETRY_DELAYS_MS = [250, 750];
export const MAX_RETRY_DELAY_MS = 5_000;

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

function dashboardDataError(code, message, cause) {
  const error = new Error(message, cause ? { cause } : undefined);
  error.dashboardDataErrorCode = code;
  return error;
}

function isRetryableStatus(status) {
  return status === 408 || status === 429 || status >= 500;
}

function retryAfterDelayMs(response) {
  const value = response.headers?.get?.('retry-after')?.trim();
  if (!value) return null;
  const seconds = Number(value);
  const delay = Number.isFinite(seconds)
    ? seconds * 1_000
    : Date.parse(value) - Date.now();
  if (!Number.isFinite(delay)) return null;
  return Math.min(MAX_RETRY_DELAY_MS, Math.max(0, delay));
}

function retryDelayMs(baseDelayMs, response, random) {
  const retryAfter = response ? retryAfterDelayMs(response) : null;
  if (retryAfter !== null) return retryAfter;
  return baseDelayMs * (0.5 + random());
}

async function waitForRetry(delayMs, signal) {
  throwIfCancelled(signal);
  if (!Number.isFinite(delayMs) || delayMs <= 0) return;
  await new Promise((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(new LoadCancelledError());
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, delayMs);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

async function fetchJsonResource(url, {
  fetchImpl,
  signal,
  timeoutMs,
  cache,
  resourceType,
  retryDelaysMs,
  random,
}) {
  for (let attempt = 0; ; attempt += 1) {
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
    let cleanedUp = false;
    const cleanup = () => {
      if (cleanedUp) return;
      cleanedUp = true;
      if (timer) clearTimeout(timer);
      signal?.removeEventListener('abort', forwardAbort);
    };
    let response;
    try {
      response = await fetchImpl(String(url), {
        signal: controller.signal,
        ...(cache ? { cache } : {}),
      });
    } catch (error) {
      cleanup();
      if (signal?.aborted) throw new LoadCancelledError();
      const failure = timedOut
        ? new Error(`Timed out after ${timeoutMs} ms loading ${resourceType} ${url}`, { cause: error })
        : error;
      if (attempt >= retryDelaysMs.length) {
        throw dashboardDataError(
          'unavailable',
          `Unable to reach ${resourceType} ${url}: ${failure.message}`,
          failure,
        );
      }
      await waitForRetry(retryDelayMs(retryDelaysMs[attempt], null, random), signal);
      continue;
    }

    if (!response.ok) {
      const failure = new Error(
        `Unable to load ${resourceType} ${url} (${response.status} ${response.statusText})`,
      );
      if (!isRetryableStatus(response.status) || attempt >= retryDelaysMs.length) {
        cleanup();
        const code = response.status === 404
          && (resourceType === 'dashboard metadata' || resourceType === 'dashboard data index')
          ? 'missing'
          : isRetryableStatus(response.status)
            ? 'unavailable'
            : 'invalid';
        throw dashboardDataError(code, failure.message, failure);
      }
      cleanup();
      await waitForRetry(retryDelayMs(retryDelaysMs[attempt], response, random), signal);
      continue;
    }
    const contentType = response.headers?.get?.('content-type');
    const normalizedContentType = contentType?.toLowerCase();
    if (
      normalizedContentType
      && !normalizedContentType.includes('json')
      && !normalizedContentType.startsWith('text/plain')
    ) {
      cleanup();
      if (resourceType === 'dashboard metadata' || resourceType === 'dashboard data index') {
        throw dashboardDataError('missing', 'No available test data');
      }
      throw dashboardDataError(
        'invalid',
        `Unable to parse ${resourceType} ${url} as JSON: received ${contentType}`,
      );
    }
    let body;
    try {
      body = await response.text();
    } catch (bodyError) {
      if (signal?.aborted) {
        cleanup();
        throw new LoadCancelledError();
      }
      const failure = timedOut
        ? new Error(
          `Timed out after ${timeoutMs} ms loading ${resourceType} ${url}`,
          { cause: bodyError },
        )
        : bodyError;
      if (attempt >= retryDelaysMs.length) {
        cleanup();
        throw dashboardDataError(
          'unavailable',
          `Unable to reach ${resourceType} ${url}: ${failure.message}`,
          failure,
        );
      }
      cleanup();
      await waitForRetry(retryDelayMs(retryDelaysMs[attempt], null, random), signal);
      continue;
    }
    try {
      return JSON.parse(body);
    } catch (parseError) {
      if (signal?.aborted) throw new LoadCancelledError();
      if (timedOut) {
        const failure = new Error(
          `Timed out after ${timeoutMs} ms loading ${resourceType} ${url}`,
          { cause: parseError },
        );
        if (attempt >= retryDelaysMs.length) {
          throw dashboardDataError(
            'unavailable',
            `Unable to reach ${resourceType} ${url}: ${failure.message}`,
            failure,
          );
        }
        cleanup();
        await waitForRetry(retryDelayMs(retryDelaysMs[attempt], null, random), signal);
        continue;
      }
      throw dashboardDataError(
        'invalid',
        `Unable to parse ${resourceType} ${url} as JSON: ${parseError.message}`,
        parseError,
      );
    } finally {
      cleanup();
    }
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
  retryDelaysMs = DEFAULT_RETRY_DELAYS_MS,
  random = Math.random,
  reloadAll = false,
  cacheGeneration = null,
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
  // Mutable documents must revalidate on every load. Runs and catalogs are immutable by
  // contract, so cached copies remain valid beyond the hosting service's freshness window. An
  // explicit reload changes every URL as well as bypassing the browser cache, which also bypasses
  // shared CDN entries that the browser cannot purge.
  const loadSignal = loadController.signal;
  const reloadToken = reloadAll ? Date.now().toString(36) : null;
  const activeCacheGeneration = reloadToken || cacheGeneration;
  const mutableRequest = {
    fetchImpl,
    signal: loadSignal,
    timeoutMs: requestTimeoutMs,
    retryDelaysMs,
    random,
    cache: 'no-store',
  };
  const [metadata, index] = await Promise.all([
    fetchJsonResource(withReloadToken(metadataUrl, reloadToken), {
      ...mutableRequest,
      resourceType: 'dashboard metadata',
    }),
    fetchJsonResource(withReloadToken(indexUrl, reloadToken), {
      ...mutableRequest,
      resourceType: 'dashboard data index',
    }),
  ]);
  if (!index || !Array.isArray(index.runFiles)) {
    throw new Error('Expected the dashboard data index to contain a runFiles array');
  }
  const immutableBaseUrl = new URL('./', indexUrl);
  const immutableRequest = {
    fetchImpl,
    signal: loadSignal,
    timeoutMs: requestTimeoutMs,
    retryDelaysMs,
    random,
    cache: reloadAll ? 'reload' : 'force-cache',
  };
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
      const run = await fetchJsonResource(
        withReloadToken(new URL(runFile, immutableBaseUrl), activeCacheGeneration),
        {
        ...immutableRequest,
        resourceType: 'run file',
        },
      );
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
      const catalog = await fetchJsonResource(
        withReloadToken(new URL(catalogPath, immutableBaseUrl), activeCacheGeneration),
        {
          ...immutableRequest,
          resourceType: 'test catalog',
        },
      );
      return { catalogPath, catalog, error: null };
    } catch (error) {
      if (isLoadCancelled(error)) throw error;
      return { catalogPath, catalog: null, error };
    }
  });

  throwIfCancelled(loadSignal);
  const unavailableResources = [
    ...runResults.map((result) => result.error),
    ...catalogResults.map((result) => result.error),
  ].filter((error) => error?.dashboardDataErrorCode === 'unavailable');
  if (unavailableResources.length > 0) {
    throw dashboardDataError(
      'unavailable',
      `Unable to load ${unavailableResources.length} published data `
      + `${unavailableResources.length === 1 ? 'resource' : 'resources'}:\n`
      + unavailableResources.map((error) => `- ${error.message}`).join('\n'),
    );
  }

  return {
    ...validatePublishedDashboardData({
    metadata,
    index,
    runs: runResults.map((result) => result.run),
    runErrors: runResults.map((result) => result.error),
    catalogs: Object.fromEntries(catalogResults.map((result) => [result.catalogPath, result.catalog])),
    catalogErrors: Object.fromEntries(catalogResults
      .filter((result) => result.error)
      .map((result) => [result.catalogPath, result.error])),
    }),
    cacheGeneration: activeCacheGeneration,
  };
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
