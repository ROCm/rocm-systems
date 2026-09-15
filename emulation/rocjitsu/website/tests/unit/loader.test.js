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
      const body = dataset.bodies.get(url);
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

  const { data, warnings } = await loadSynthetic(fetchImpl);

  expect(warnings).toEqual([]);
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

test('revalidates the mutable documents and lets immutable files use HTTP caching', async () => {
  const { fetchImpl, state } = createFetchDouble();

  await loadSynthetic(fetchImpl);

  expect(state.options.get(dataset.metadataUrl).cache).toBe('no-store');
  expect(state.options.get(dataset.indexUrl).cache).toBe('no-store');
  expect(state.options.get(`https://dashboard.test/data/${dataset.runFiles[0]}`).cache).toBeUndefined();
  expect(state.options.get(`https://dashboard.test/data/${dataset.catalogPath}`).cache).toBeUndefined();
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

test('turns a timed-out run file into a warning and keeps the rest of the history', async () => {
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

  const { data, warnings } = await loadSynthetic(fetchImpl, { requestTimeoutMs: 30 });

  expect(data.runs).toHaveLength(dataset.runCount - 1);
  expect(warnings).toHaveLength(1);
  expect(warnings[0].runFile).toBe(dataset.runFiles[7]);
  expect(warnings[0].message).toContain('Timed out after 30 ms');
  expect(warnings[0].message).toContain(stalledUrl);
});

test('turns an unrelated AbortError from one run file into a warning', async () => {
  const abortedUrl = `https://dashboard.test/data/${dataset.runFiles[7]}`;
  const { fetchImpl } = createFetchDouble({
    behavior: (url) => {
      if (url !== abortedUrl) return null;
      const error = new Error('The connection was aborted');
      error.name = 'AbortError';
      return Promise.reject(error);
    },
  });

  const { data, warnings } = await loadSynthetic(fetchImpl);

  expect(data.runs).toHaveLength(dataset.runCount - 1);
  expect(warnings).toEqual([{
    runFile: dataset.runFiles[7],
    message: 'The connection was aborted',
  }]);
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

test('names the resource in fetch and JSON-parse diagnostics', async () => {
  const missingUrl = `https://dashboard.test/data/${dataset.runFiles[1]}`;
  const unparsableUrl = `https://dashboard.test/data/${dataset.runFiles[2]}`;
  const { fetchImpl } = createFetchDouble({
    behavior: (url) => {
      if (url === missingUrl) return Promise.resolve({ ok: false, status: 404, statusText: 'Not Found' });
      if (url === unparsableUrl) {
        return Promise.resolve({
          ok: true,
          status: 200,
          statusText: 'OK',
          json: async () => { throw new SyntaxError('Unexpected token <'); },
        });
      }
      return null;
    },
  });

  const { warnings } = await loadSynthetic(fetchImpl);

  expect(warnings.map((warning) => warning.message)).toEqual([
    `Unable to load run file ${missingUrl} (404 Not Found)`,
    `Unable to parse run file ${unparsableUrl} as JSON: Unexpected token <`,
  ]);
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

  await expect(loadSynthetic(fetchImpl)).rejects.toThrow(
    `Unable to load dashboard data index ${dataset.indexUrl} (503 Service Unavailable)`,
  );
});

test('matches the result an unbounded loader produces', async () => {
  const { fetchImpl } = createFetchDouble();

  const bounded = await loadSynthetic(fetchImpl, { concurrency: MAX_CONCURRENT_RUN_REQUESTS });
  const unbounded = await loadSynthetic(fetchImpl, { concurrency: dataset.runCount });

  expect(JSON.stringify(bounded.data)).toBe(JSON.stringify(unbounded.data));
  expect(bounded.warnings).toEqual(unbounded.warnings);
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
