import { beforeEach, expect, test, vi } from 'vitest';
import {
  MAX_CONCURRENT_RUN_REQUESTS,
  isLoadCancelled,
  loadDashboardDataFiles,
} from '../../src/data/dashboardData.js';
import { createSyntheticDataset } from '../fixtures/syntheticDataset.js';

let dataset;

function jsonResponse(body) {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    text: async () => JSON.stringify(body),
    json: async () => structuredClone(body),
  };
}

// A fetch double that records concurrency, request options, and per-URL behavior overrides.
function createFetchDouble({ delayMs = 0, behavior = () => null } = {}) {
  const state = { active: 0, peak: 0, urls: [], options: new Map() };
  const fetchImpl = async (url, options = {}) => {
    state.active += 1;
    state.peak = Math.max(state.peak, state.active);
    state.urls.push(url);
    state.options.set(url, options);
    try {
      const override = behavior(url, options);
      if (override) return await override;
      if (delayMs) await new Promise((resolve) => setTimeout(resolve, delayMs));
      const resourceUrl = new URL(url);
      resourceUrl.search = '';
      const body = dataset.bodies.get(resourceUrl.href);
      if (!body) return { ok: false, status: 404, statusText: 'Not Found', json: async () => ({}) };
      return jsonResponse(body);
    } finally {
      state.active -= 1;
    }
  };
  return { fetchImpl, state };
}

function loadSynthetic(fetchImpl, overrides = {}) {
  return loadDashboardDataFiles({
    metadataUrl: dataset.metadataUrl,
    indexUrl: dataset.indexUrl,
    fetch: fetchImpl,
    ...overrides,
  });
}

beforeEach(() => {
  dataset = createSyntheticDataset(500);
});

test('loads 500 runs in index order with at most eight run requests in flight', async () => {
  const { fetchImpl, state } = createFetchDouble({ delayMs: 1 });

  const { data } = await loadSynthetic(fetchImpl);

  expect(data.runs).toHaveLength(dataset.runCount);
  expect(data.runs.map((run) => run.runId)).toEqual(
    dataset.runFiles.map((runFile) => runFile.replace(/^runs\/|\.json$/g, '')),
  );
  expect(state.peak).toBeLessThanOrEqual(MAX_CONCURRENT_RUN_REQUESTS);
  expect(state.urls).toHaveLength(dataset.runCount + 3);
});

test('honors a caller-supplied concurrency limit', async () => {
  const { fetchImpl, state } = createFetchDouble({ delayMs: 1 });

  await loadSynthetic(fetchImpl, { concurrency: 3 });

  expect(state.peak).toBeLessThanOrEqual(3);
});

test('revalidates mutable documents and reuses cached immutable files', async () => {
  const { fetchImpl, state } = createFetchDouble();

  await loadSynthetic(fetchImpl);

  expect(state.options.get(dataset.metadataUrl).cache).toBe('no-store');
  expect(state.options.get(dataset.indexUrl).cache).toBe('no-store');
  expect(state.options.get(`https://dashboard.test/data/${dataset.runFiles[0]}`).cache).toBe('force-cache');
  expect(state.options.get(`https://dashboard.test/data/${dataset.catalogPath}`).cache).toBe('force-cache');
});

test('reloads a fresh index before every file it names and saves a new cache generation', async () => {
  const freshRunFiles = dataset.runFiles.slice(-2);
  const { fetchImpl, state } = createFetchDouble({
    behavior: (url) => {
      const requestUrl = new URL(url);
      if (
        requestUrl.pathname.endsWith('/index.json')
        && requestUrl.searchParams.has('reload')
      ) {
        return jsonResponse({
          generatedAt: '2027-06-02T00:00:00.000Z',
          runFiles: freshRunFiles,
        });
      }
      return null;
    },
  });

  const { data, cacheGeneration } = await loadSynthetic(fetchImpl, { reloadAll: true });

  const requestUrls = state.urls.map((url) => new URL(url));
  const reloadTokens = new Set(requestUrls.map((url) => url.searchParams.get('reload')));
  const indexRequest = requestUrls.findIndex((url) => url.pathname.endsWith('/index.json'));
  const firstRunRequest = requestUrls.findIndex((url) => url.pathname.includes('/runs/'));

  expect(data.runs.map((run) => `${run.runId}.json`)).toEqual(
    freshRunFiles.map((runFile) => runFile.replace('runs/', '')),
  );
  expect(requestUrls.filter((url) => url.pathname.includes('/runs/')).map(
    (url) => url.pathname.replace('/data/', ''),
  )).toEqual(freshRunFiles);
  expect([...reloadTokens]).toHaveLength(1);
  expect([...reloadTokens][0]).toBeTruthy();
  expect(cacheGeneration).toBe([...reloadTokens][0]);
  expect(indexRequest).toBeGreaterThanOrEqual(0);
  expect(firstRunRequest).toBeGreaterThan(indexRequest);
  expect([...state.options].every(([url, options]) => (
    url.includes('/metadata.json') || url.includes('/index.json')
      ? options.cache === 'no-store'
      : options.cache === 'reload'
  ))).toBe(true);
});

test('reuses a saved cache generation on the next normal load', async () => {
  dataset = createSyntheticDataset(2);
  const { fetchImpl, state } = createFetchDouble();

  await loadSynthetic(fetchImpl, { cacheGeneration: 'saved-generation' });

  const requestUrls = state.urls.map((url) => new URL(url));
  const mutableRequests = requestUrls.filter((url) => (
    url.pathname.endsWith('/metadata.json') || url.pathname.endsWith('/index.json')
  ));
  const immutableRequests = requestUrls.filter((url) => !mutableRequests.includes(url));

  expect(mutableRequests.every((url) => !url.search)).toBe(true);
  expect(immutableRequests.every(
    (url) => url.searchParams.get('reload') === 'saved-generation',
  )).toBe(true);
  expect(immutableRequests.every(
    (url) => state.options.get(url.href).cache === 'force-cache',
  )).toBe(true);
});

test('retries transient HTTP failures before failing the load', async () => {
  let remainingFailures = 2;
  const { fetchImpl, state } = createFetchDouble({
    behavior: (url) => {
      if (url !== dataset.indexUrl || remainingFailures === 0) return null;
      remainingFailures -= 1;
      return Promise.resolve({ ok: false, status: 503, statusText: 'Service Unavailable' });
    },
  });

  const { data } = await loadSynthetic(fetchImpl, { retryDelaysMs: [0, 0] });

  expect(data.runs).toHaveLength(dataset.runCount);
  expect(state.urls.filter((url) => url === dataset.indexUrl)).toHaveLength(3);
});

test('retries transient response-body failures before failing the load', async () => {
  let remainingFailures = 1;
  const { fetchImpl, state } = createFetchDouble({
    behavior: (url) => {
      if (url !== dataset.indexUrl || remainingFailures === 0) return null;
      remainingFailures -= 1;
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: async () => {
          throw new TypeError('The connection closed while reading the body');
        },
      });
    },
  });

  const { data } = await loadSynthetic(fetchImpl, { retryDelaysMs: [0] });

  expect(data.runs).toHaveLength(dataset.runCount);
  expect(state.urls.filter((url) => url === dataset.indexUrl)).toHaveLength(2);
});

test('honors Retry-After without applying jitter', async () => {
  let failIndex = true;
  const random = vi.fn(() => {
    throw new Error('jitter should not be used');
  });
  const { fetchImpl } = createFetchDouble({
    behavior: (url) => {
      if (url !== dataset.indexUrl || !failIndex) return null;
      failIndex = false;
      return Promise.resolve({
        ok: false,
        status: 429,
        statusText: 'Too Many Requests',
        headers: { get: (name) => (name === 'retry-after' ? '0' : null) },
      });
    },
  });

  const { data } = await loadSynthetic(fetchImpl, {
    retryDelaysMs: [250],
    random,
  });

  expect(data.runs).toHaveLength(dataset.runCount);
  expect(random).not.toHaveBeenCalled();
});

test('reports determinate progress for every run file exactly once', async () => {
  const { fetchImpl } = createFetchDouble();
  const updates = [];

  await loadSynthetic(fetchImpl, { onProgress: (update) => updates.push(update) });

  expect(updates[0]).toEqual({ loaded: 0, total: dataset.runCount });
  expect(updates).toHaveLength(dataset.runCount + 1);
  expect(updates.at(-1)).toEqual({ loaded: dataset.runCount, total: dataset.runCount });
  expect(updates.every((update, index) => update.loaded === index)).toBe(true);
});

test('fails closed when a run file times out', async () => {
  const stalledUrl = `https://dashboard.test/data/${dataset.runFiles[7]}`;
  const { fetchImpl } = createFetchDouble({
    behavior: (url, options) => (url === stalledUrl
      ? new Promise((_, reject) => {
        options.signal.addEventListener('abort', () => {
          const error = new Error('aborted');
          error.name = 'AbortError';
          reject(error);
        });
      })
      : null),
  });

  await expect(loadSynthetic(fetchImpl, {
    requestTimeoutMs: 30,
    retryDelaysMs: [],
  })).rejects.toThrow(
    new RegExp(`Timed out after 30 ms[\\s\\S]*${stalledUrl.replaceAll('/', '\\/')}`),
  );
});

test('keeps the request timeout active while reading a response body', async () => {
  const stalledUrl = `https://dashboard.test/data/${dataset.runFiles[7]}`;
  const { fetchImpl } = createFetchDouble({
    behavior: (url, options) => {
      if (url !== stalledUrl) return null;
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: 'OK',
        headers: { get: () => 'application/json' },
        text: () => new Promise((_, reject) => {
          options.signal.addEventListener('abort', () => {
            const error = new Error('body read aborted');
            error.name = 'AbortError';
            reject(error);
          });
        }),
      });
    },
  });

  await expect(loadSynthetic(fetchImpl, {
    requestTimeoutMs: 30,
    retryDelaysMs: [],
  })).rejects.toThrow(
    new RegExp(`Timed out after 30 ms[\\s\\S]*${stalledUrl.replaceAll('/', '\\/')}`),
  );
});

test('fails closed for an unrelated AbortError from one run file', async () => {
  const abortedUrl = `https://dashboard.test/data/${dataset.runFiles[7]}`;
  const { fetchImpl } = createFetchDouble({
    behavior: (url) => {
      if (url !== abortedUrl) return null;
      const error = new Error('The connection was aborted');
      error.name = 'AbortError';
      return Promise.reject(error);
    },
  });

  await expect(loadSynthetic(fetchImpl, { retryDelaysMs: [] })).rejects.toThrow(
    `Unable to reach run file ${abortedUrl}: The connection was aborted`,
  );
});

test('applies a deadline to the whole load and surfaces a retryable error', async () => {
  const { fetchImpl } = createFetchDouble({
    behavior: (url, options) => {
      if (!url.includes('/runs/')) return null;
      return new Promise((_, reject) => {
        options.signal.addEventListener('abort', () => {
          const error = new Error('aborted');
          error.name = 'AbortError';
          reject(error);
        });
      });
    },
  });

  const failure = await loadSynthetic(fetchImpl, {
    loadTimeoutMs: 30,
    requestTimeoutMs: 1_000,
  }).catch((error) => error);

  expect(isLoadCancelled(failure)).toBe(false);
  expect(failure).toHaveProperty('message', 'Timed out after 30 ms loading dashboard data');
});

test('keeps the whole-load deadline active while reading response bodies', async () => {
  const stalledUrl = `https://dashboard.test/data/${dataset.runFiles[7]}`;
  const { fetchImpl } = createFetchDouble({
    behavior: (url, options) => {
      if (url !== stalledUrl) return null;
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: 'OK',
        headers: { get: () => 'application/json' },
        text: () => new Promise((_, reject) => {
          options.signal.addEventListener('abort', () => {
            const error = new Error('body read aborted');
            error.name = 'AbortError';
            reject(error);
          });
        }),
      });
    },
  });

  const failure = await loadSynthetic(fetchImpl, {
    loadTimeoutMs: 30,
    requestTimeoutMs: 1_000,
  }).catch((error) => error);

  expect(isLoadCancelled(failure)).toBe(false);
  expect(failure).toHaveProperty('message', 'Timed out after 30 ms loading dashboard data');
});

test('does not retry missing or malformed immutable resources', async () => {
  const missingUrl = `https://dashboard.test/data/${dataset.runFiles[1]}`;
  const unparsableUrl = `https://dashboard.test/data/${dataset.runFiles[2]}`;
  const { fetchImpl, state } = createFetchDouble({
    behavior: (url) => {
      if (url === missingUrl) return Promise.resolve({ ok: false, status: 404, statusText: 'Not Found' });
      if (url === unparsableUrl) {
        return Promise.resolve({
          ok: true,
          status: 200,
          statusText: 'OK',
          text: async () => '{ invalid json',
        });
      }
      return null;
    },
  });

  const error = await loadSynthetic(fetchImpl).catch((loadError) => loadError);

  expect(error.message).toContain(`Unable to load run file ${missingUrl} (404 Not Found)`);
  expect(error.message).toContain(
    `Unable to parse run file ${unparsableUrl} as JSON:`,
  );
  expect(state.urls.filter((url) => url === missingUrl)).toHaveLength(1);
  expect(state.urls.filter((url) => url === unparsableUrl)).toHaveLength(1);
});

test('an external abort cancels the whole load instead of skipping runs', async () => {
  const controller = new AbortController();
  const { fetchImpl, state } = createFetchDouble({
    behavior: (url) => {
      if (url.includes('/runs/') && state.urls.length > 20) controller.abort();
      return null;
    },
  });

  const failure = await loadSynthetic(fetchImpl, { signal: controller.signal }).catch((error) => error);

  expect(isLoadCancelled(failure)).toBe(true);
  expect(state.urls.length).toBeLessThan(dataset.runCount);
});

test('a failing index is fatal rather than a warning', async () => {
  const { fetchImpl } = createFetchDouble({
    behavior: (url) => (url === dataset.indexUrl
      ? Promise.resolve({ ok: false, status: 503, statusText: 'Service Unavailable' })
      : null),
  });

  await expect(loadSynthetic(fetchImpl, { retryDelaysMs: [] })).rejects.toThrow(
    `Unable to load dashboard data index ${dataset.indexUrl} (503 Service Unavailable)`,
  );
});

test('treats an HTML fallback for the data index as missing test data', async () => {
  const parseHtml = vi.fn();
  const { fetchImpl } = createFetchDouble({
    behavior: (url) => (url === dataset.indexUrl
      ? Promise.resolve({
        ok: true,
        status: 200,
        statusText: 'OK',
        headers: { get: () => 'text/html; charset=utf-8' },
        json: parseHtml,
      })
      : null),
  });

  await expect(loadSynthetic(fetchImpl)).rejects.toThrow('No available test data');
  expect(parseHtml).not.toHaveBeenCalled();
});

test('accepts JSON served as text/plain by a raw file host', async () => {
  const { fetchImpl } = createFetchDouble({
    behavior: (url) => {
      const body = dataset.bodies.get(url);
      if (!body) return null;
      return Promise.resolve({
        ...jsonResponse(body),
        headers: { get: () => 'text/plain; charset=utf-8' },
      });
    },
  });

  const { data } = await loadSynthetic(fetchImpl);

  expect(data.runs).toHaveLength(dataset.runCount);
});

test('matches the result an unbounded loader produces', async () => {
  const { fetchImpl } = createFetchDouble();

  const bounded = await loadSynthetic(fetchImpl, { concurrency: MAX_CONCURRENT_RUN_REQUESTS });
  const unbounded = await loadSynthetic(fetchImpl, { concurrency: dataset.runCount });

  expect(JSON.stringify(bounded.data)).toBe(JSON.stringify(unbounded.data));
});

test('cancels in-flight requests when the caller aborts', async () => {
  const controller = new AbortController();
  const abortedSignals = [];
  const { fetchImpl, state } = createFetchDouble({
    behavior: (url, options) => {
      if (!url.includes('/runs/')) return null;
      return new Promise((_, reject) => {
        options.signal.addEventListener('abort', () => {
          abortedSignals.push(url);
          const error = new Error('aborted');
          error.name = 'AbortError';
          reject(error);
        });
      });
    },
  });

  const pending = loadSynthetic(fetchImpl, { signal: controller.signal }).catch((error) => error);
  await vi.waitFor(() => expect(state.active).toBe(MAX_CONCURRENT_RUN_REQUESTS));
  controller.abort();

  expect(isLoadCancelled(await pending)).toBe(true);
  expect(abortedSignals).toHaveLength(MAX_CONCURRENT_RUN_REQUESTS);
});
