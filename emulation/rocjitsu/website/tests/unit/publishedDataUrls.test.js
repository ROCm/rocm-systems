import { expect, test } from 'vitest';
import { resolvePublishedDataUrls } from '../../src/data/publishedDataUrls.js';

const pagesBase = 'https://example.github.io/rocm-systems/rocjitsu-dashboard/';
const dataBranchBase = (
  'https://raw.githubusercontent.com/ROCm/rocm-systems/'
  + 'refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/'
);

test('uses the configured data-branch URL for metadata and index', () => {
  const urls = resolvePublishedDataUrls({
    dataBaseUrl: dataBranchBase,
    baseURI: pagesBase,
  });

  expect(urls.metadataUrl).toBe(`${dataBranchBase}metadata.json`);
  expect(urls.indexUrl).toBe(`${dataBranchBase}index.json`);
});

test('falls back to same-origin data next to the app when unset', () => {
  const urls = resolvePublishedDataUrls({
    dataBaseUrl: '',
    assetBaseUrl: './',
    baseURI: pagesBase,
  });

  expect(urls.metadataUrl).toBe(`${pagesBase}data/metadata.json`);
  expect(urls.indexUrl).toBe(`${pagesBase}data/index.json`);
});

test('adds a trailing slash so a branch root is treated as a directory', () => {
  const urls = resolvePublishedDataUrls({
    dataBaseUrl: dataBranchBase.replace(/\/$/, ''),
    baseURI: pagesBase,
  });

  expect(urls.metadataUrl).toBe(`${dataBranchBase}metadata.json`);
});
