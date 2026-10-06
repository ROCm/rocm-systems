import { readFileSync } from 'node:fs';

const fixtureRoot = new URL('../../fixtures/data/', import.meta.url);
const readJson = (name) => JSON.parse(readFileSync(new URL(name, fixtureRoot), 'utf8'));

export async function serveThreadingDataset(page, modes = ['default', 'single']) {
  const index = readJson('index.json');
  const original = readJson(index.runFiles[0]);
  const originalCatalog = readJson(original.testCatalog);
  const resources = new Map();
  const runFiles = [];
  for (const mode of modes) {
    const target = mode === 'default' ? 'gfx1250' : 'gfx950';
    const ids = originalCatalog.targets[target].filter((id) => mode === 'default'
      || originalCatalog.tests.find((test) => test.id === id).suite === 'TensileLite');
    const catalogPath = `test-catalogs/threading-${mode}.json`;
    resources.set(catalogPath, { id: `threading-${mode}`, tests: originalCatalog.tests.filter((test) => ids.includes(test.id)), targets: { [target]: ids } });
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const run = structuredClone(original);
      run.id = `threading-${mode}-${attempt}`;
      run.comparisonId = run.id;
      run.threadingMode = mode;
      run.testCatalog = catalogPath;
      run.source.commit = String(attempt + 1).repeat(40);
      run.source.committedAt = `2026-09-0${attempt + 1}T00:00:00.000Z`;
      run.execution.completedAt = `2026-09-0${attempt + 1}T01:00:00.000Z`;
      run.targets = run.targets.filter((group) => group.id === target).map((group) => ({
        ...group,
        results: group.results.filter((result) => ids.includes(result.testId)).map((result) => ({
          ...result,
          durationSeconds: mode === 'single' ? 500 : 10,
          status: 'completed', error: null,
        })),
      }));
      const path = `runs/${run.id}.json`;
      runFiles.push(path);
      resources.set(path, run);
    }
  }
  resources.set('index.json', { ...index, runFiles });
  resources.set('metadata.json', readJson('metadata.json'));
  await page.route('**/data/**', (route) => {
    const path = new URL(route.request().url()).pathname.split('/data/')[1];
    const resource = resources.get(path);
    return resource ? route.fulfill({ json: resource }) : route.continue();
  });
}

export async function switchThreading(page, label) {
  await page.getByRole('combobox', { name: 'Threading', exact: true }).click();
  for (const option of ['Default', 'Single-thread']) {
    const item = page.getByRole('option', { name: option, exact: true });
    if (await item.count() && (await item.getAttribute('aria-selected') === 'true') !== (option === label)) {
      await item.click();
    }
  }
  await page.keyboard.press('Escape');
}
