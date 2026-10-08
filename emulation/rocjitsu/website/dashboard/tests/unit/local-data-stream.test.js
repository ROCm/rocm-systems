import { PassThrough, Writable } from 'node:stream';
import { fileURLToPath } from 'node:url';
import { expect, test, vi } from 'vitest';
import { createReadStream } from 'node:fs';
import { createDashboardDataMiddleware } from '../../scripts/dashboard-data-dir.mjs';

vi.mock('node:fs', async (importOriginal) => ({
  ...await importOriginal(),
  createReadStream: vi.fn(),
}));

const directory = fileURLToPath(new URL('../fixtures/data/', import.meta.url));

class Response extends Writable {
  headersSent = false;
  statusCode = 200;
  setHeader() {}
  _write(_chunk, _encoding, callback) {
    this.headersSent = true;
    callback();
  }
}

async function startStream() {
  const source = new PassThrough();
  const response = new Response();
  const started = new Promise((resolve) => {
    createReadStream.mockImplementationOnce(() => {
      resolve();
      return source;
    });
  });
  createDashboardDataMiddleware(directory)({ url: '/data/metadata.json' }, response, () => {});
  await started;
  return { source, response };
}

test('a read failure after headers destroys the response instead of completing partial JSON', async () => {
  const { source, response } = await startStream();
  source.write('{"partial":');
  expect(response.headersSent).toBe(true);
  const closed = new Promise((resolve) => response.once('close', resolve));
  source.destroy(new Error('Read failed after headers'));
  await closed;
  expect(response.destroyed).toBe(true);
  expect(response.writableFinished).toBe(false);
});

test('a client disconnect destroys its outstanding file stream', async () => {
  const { source, response } = await startStream();
  const closed = new Promise((resolve) => source.once('close', resolve));
  response.destroy();
  await closed;
  expect(source.destroyed).toBe(true);
});

test('a response write failure destroys its outstanding file stream', async () => {
  const { source, response } = await startStream();
  const closed = new Promise((resolve) => source.once('close', resolve));
  response.destroy(new Error('Client write failed'));
  await closed;
  expect(source.destroyed).toBe(true);
});
