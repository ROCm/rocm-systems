import { Writable } from 'node:stream';
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from 'vitest';
import {
  createDashboardDataMiddleware,
  parseServeDataArgs,
  resolveDashboardDataDirectory,
} from '../../scripts/dashboard-data-dir.mjs';

const fixtureDataDirectory = path.resolve(fileURLToPath(new URL('../fixtures/data/', import.meta.url)));
const fixtureRoot = path.resolve(fileURLToPath(new URL('../fixtures/', import.meta.url)));

test('resolves a data directory or a parent that contains data/', () => {
  expect(resolveDashboardDataDirectory(fixtureDataDirectory)).toBe(fixtureDataDirectory);
  expect(resolveDashboardDataDirectory(fixtureRoot)).toBe(fixtureDataDirectory);
});

test('rejects a directory that is not dashboard data', () => {
  expect(() => resolveDashboardDataDirectory(fileURLToPath(new URL('../', import.meta.url))))
    .toThrow(/Expected a dashboard data directory/);
  expect(() => parseServeDataArgs(['--host', '127.0.0.1'])).toThrow(/Usage: npm run dev:data/);
});

test('parses a data directory and leftover Vite arguments', () => {
  expect(parseServeDataArgs([
    fixtureDataDirectory,
    '--host',
    '127.0.0.1',
  ])).toEqual({
    directory: fixtureDataDirectory,
    preview: false,
    viteArgs: ['--host', '127.0.0.1'],
  });
  expect(parseServeDataArgs(['--preview', fixtureRoot, '--port', '4173'])).toMatchObject({
    directory: fixtureRoot,
    preview: true,
    viteArgs: ['--port', '4173'],
  });
});

class ResponseStub extends Writable {
  constructor() {
    super();
    this.statusCode = 200;
    this.headers = {};
    this.chunks = [];
  }

  setHeader(name, value) {
    this.headers[name] = value;
  }

  _write(chunk, _encoding, callback) {
    this.chunks.push(Buffer.from(chunk));
    callback();
  }
}

function request(middleware, url) {
  return new Promise((resolve) => {
    const res = new ResponseStub();
    res.on('finish', () => resolve({
      status: res.statusCode,
      body: Buffer.concat(res.chunks),
      headers: res.headers,
    }));
    middleware({ url }, res, () => {
      res.statusCode = 404;
      res.end();
    });
  });
}

test('serves JSON from the mounted /data/ path and blocks escapes', async () => {
  const middleware = createDashboardDataMiddleware(fixtureDataDirectory);

  const metadata = await request(middleware, '/data/metadata.json');
  expect(metadata.status).toBe(200);
  expect(metadata.headers['Content-Type']).toMatch(/application\/json/);
  expect(JSON.parse(metadata.body.toString())).toMatchObject({ schemaVersion: 1 });

  const escaped = await request(middleware, '/data/../README.md');
  expect(escaped.status).toBe(400);

  const encodedEscape = await request(middleware, '/data/%2e%2e/README.md');
  expect(encodedEscape.status).toBe(400);

  const malformedEncoding = await request(middleware, '/data/%');
  expect(malformedEncoding.status).toBe(400);
  expect(malformedEncoding.body).toHaveLength(0);

  const missing = await request(middleware, '/data/runs/does-not-exist.json');
  expect(missing.status).toBe(404);
});

test('blocks symlinks that resolve outside the data directory', async () => {
  const temporaryDirectory = await mkdtemp(path.join(tmpdir(), 'rocjitsu-dashboard-data-'));
  const dataDirectory = path.join(temporaryDirectory, 'data');
  const outsideFile = path.join(temporaryDirectory, 'outside.json');

  try {
    await mkdir(dataDirectory);
    await writeFile(outsideFile, '{"secret":true}');
    await symlink(outsideFile, path.join(dataDirectory, 'linked.json'));

    const response = await request(
      createDashboardDataMiddleware(dataDirectory),
      '/data/linked.json',
    );
    expect(response.status).toBe(400);
    expect(response.body).toHaveLength(0);
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
});
