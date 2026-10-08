import { Writable } from 'node:stream';
import { execFileSync } from 'node:child_process';
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

test('a file removed after stat fails only its request and leaves the server alive', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'dashboard-stream-'));
  try {
    await writeFile(path.join(directory, 'removed.json'), '{}');
    await writeFile(path.join(directory, 'healthy.json'), '{"healthy":true}');
    const middlewareUrl = new URL('../../scripts/dashboard-data-dir.mjs', import.meta.url).href;
    const output = execFileSync(process.execPath, ['--input-type=module', '--eval', `
      import fs from 'node:fs';
      import { syncBuiltinESMExports } from 'node:module';
      import { createServer } from 'node:http';
      import { once } from 'node:events';
      const originalStat = fs.stat;
      // Force a real open failure after the middleware's successful stat.
      fs.stat = (file, callback) => originalStat(file, (error, info) => {
        if (!error && file.endsWith('/removed.json')) fs.unlinkSync(file);
        callback(error, info);
      });
      syncBuiltinESMExports();
      const { createDashboardDataMiddleware } = await import(${JSON.stringify(middlewareUrl)});
      const middleware = createDashboardDataMiddleware(${JSON.stringify(directory)});
      const server = createServer((req, res) => middleware(req, res, () => res.end()));
      server.listen(0, '127.0.0.1');
      await once(server, 'listening');
      try {
        const base = 'http://127.0.0.1:' + server.address().port;
        const failed = await fetch(base + '/data/removed.json');
        await failed.text();
        const healthy = await fetch(base + '/data/healthy.json');
        console.log(JSON.stringify({ failed: failed.status, healthy: healthy.status, body: await healthy.json() }));
      } finally {
        server.closeAllConnections();
        await new Promise((resolve) => server.close(resolve));
      }
    `], { encoding: 'utf8', timeout: 10_000 });
    expect(JSON.parse(output)).toEqual({ failed: 500, healthy: 200, body: { healthy: true } });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
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
  expect(JSON.parse(metadata.body.toString())).toMatchObject({ schemaVersion: 2 });

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
