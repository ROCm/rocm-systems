// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Verify repository counts, cache expiry, and graceful network/storage failures.
// Usage: node --test tests/repository-stats.test.cjs

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(
  path.join(__dirname, '../assets/repository-stats.js'), 'utf8',
);
const now = 1_800_000_000_000;
const cacheKey = 'rocjitsu-repository-stats';
const counts = { stargazers_count: 1234, forks_count: 0 };

async function run({ cache = null, response, storageThrows = false, initial = {} } = {}) {
  const elements = ['header', 'mobile'].flatMap(context =>
    ['stargazers_count', 'forks_count'].map(key => ({
      context,
      dataset: { repositoryStat: key },
      textContent: initial[key] ?? '—',
      title: 'Count unavailable',
      attributes: {},
      setAttribute(name, value) { this.attributes[name] = value; },
    })),
  );
  const requests = [];
  const writes = [];
  const result = vm.runInNewContext(source, {
    Date: { now: () => now },
    document: {
      querySelectorAll(selector) {
        assert.equal(selector, '[data-repository-stat]');
        return elements;
      },
    },
    sessionStorage: {
      getItem(key) {
        if (storageThrows) throw new Error('Storage disabled');
        assert.equal(key, cacheKey);
        return cache;
      },
      setItem(key, value) {
        if (storageThrows) throw new Error('Storage disabled');
        writes.push({ key, value: JSON.parse(value) });
      },
    },
    async fetch(url) {
      requests.push(url);
      if (response instanceof Error) throw response;
      return response || { ok: true, json: async () => counts };
    },
  });
  assert.equal(typeof result?.then, 'function', 'The loader must be awaitable');
  await result;
  return { elements, requests, writes };
}

function assertRendered(elements) {
  assert.deepEqual(elements.map(element => element.textContent), [
    '1,234', '0', '1,234', '0',
  ]);
  for (const element of elements) {
    const expected = element.dataset.repositoryStat === 'stargazers_count'
      ? 'Stars: 1,234 · ROCm/rocm-systems'
      : 'Forks: 0 · ROCm/rocm-systems';
    assert.equal(element.title, expected);
    assert.equal(element.attributes['aria-label'], expected);
  }
}

test('one request updates desktop/mobile counts, including zero, and caches them', async () => {
  const { elements, requests, writes } = await run();
  assertRendered(elements);
  assert.deepEqual(requests, ['https://api.github.com/repos/ROCm/rocm-systems']);
  assert.deepEqual(writes, [{ key: cacheKey, value: { time: now, data: counts } }]);
});

for (const [name, response] of [
  ['network rejection', new Error('Offline')],
  ['HTTP failure', { ok: false, json: async () => counts }],
  ['malformed JSON', { ok: true, json: async () => { throw new Error('Invalid JSON'); } }],
  ['partial counts', { ok: true, json: async () => ({ stargazers_count: 1234 }) }],
  ['invalid counts', { ok: true, json: async () => ({ stargazers_count: -1, forks_count: '0' }) }],
]) {
  test(`${name} preserves both placeholders without caching`, async () => {
    const { elements, requests, writes } = await run({ response });
    assert.equal(requests.length, 1);
    assert.deepEqual(writes, []);
    for (const element of elements) {
      assert.equal(element.textContent, '—');
      assert.equal(element.title, 'Count unavailable');
      assert.deepEqual(element.attributes, {});
    }
  });
}

test('fresh cache renders both contexts without a request', async () => {
  const cache = JSON.stringify({ time: now - 60_000, data: counts });
  const { elements, requests, writes } = await run({ cache });
  assertRendered(elements);
  assert.deepEqual(requests, []);
  assert.deepEqual(writes, []);
});

test('expired counts remain visible when the refresh fails', async () => {
  const cache = JSON.stringify({ time: now - 16 * 60 * 1000, data: counts });
  const { elements, requests, writes } = await run({ cache, response: new Error('Offline') });
  assertRendered(elements);
  assert.equal(requests.length, 1);
  assert.deepEqual(writes, []);
});

test('server-rendered counts survive a failed browser request without a cache', async () => {
  const initial = { stargazers_count: '516', forks_count: '425' };
  const { elements, requests, writes } = await run({ initial, response: new Error('Blocked') });
  assert.deepEqual(elements.map(element => element.textContent), ['516', '425', '516', '425']);
  assert.equal(requests.length, 1);
  assert.deepEqual(writes, []);
});

for (const [name, cache] of [
  ['expired', JSON.stringify({ time: now - 15 * 60 * 1000, data: counts })],
  ['future timestamp', JSON.stringify({ time: now + 1000, data: counts })],
  ['malformed', '{broken'],
  ['partial', JSON.stringify({ time: now, data: { forks_count: 0 } })],
]) {
  test(`${name} cache is replaced from the API`, async () => {
    const { elements, requests, writes } = await run({ cache });
    assertRendered(elements);
    assert.equal(requests.length, 1);
    assert.deepEqual(writes, [{ key: cacheKey, value: { time: now, data: counts } }]);
  });
}

test('unavailable storage does not prevent fetching or rendering', async () => {
  const { elements, requests } = await run({ storageThrows: true });
  assertRendered(elements);
  assert.equal(requests.length, 1);
});
