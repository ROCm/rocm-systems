import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import RunMetadataDiff from '../../src/components/compare/RunMetadataDiff.jsx';

it('compares relevant configuration/environment facts without noisy per-execution bookkeeping rows', () => {
  const run = {
    runId: 'fictional-reference', tests: [], catalogId: 'fictional-catalog',
    machineId: 'fictional-machine', configurations: [{ target: 'gfx1250', threadingMode: 'MT' }],
    provenance: { rocjitsuCommitSha: 'a'.repeat(40) },
    timestamp: '2026-10-01T12:00:00Z',
    trigger: 'schedule', environmentId: 'fictional-env-1',
    environment: [{ key: 'sdk', label: 'ROCm SDK', value: 'fictional-old' }],
  };
  const candidate = { ...run, runId: 'fictional-candidate',
    environmentId: 'fictional-env-2', timestamp: '2026-10-02T12:00:00Z',
    environment: [{ key: 'sdk', label: 'ROCm SDK', value: 'fictional-new' }],
  };
  const html = renderToStaticMarkup(createElement(RunMetadataDiff, {
    baseline: run, candidate, filters: { targets: ['gfx1250'], modes: ['MT'], suites: [] },
  }));
  for (const key of ['runId', 'commit', 'message', 'branch', 'base', 'pullRequest', 'commitTime', 'runTime', 'trigger', 'plugin', 'workflow', 'comparisonId', 'environmentId']) {
    expect(html.includes(`data-testid="metadata-row-${key}"`), `${key} is bookkeeping, not a comparison row`).toBe(false);
  }
  for (const key of ['coverage', 'catalog', 'machine', 'configurations', 'sdk']) {
    expect(html.includes(`data-testid="metadata-row-${key}"`), `${key} remains comparable`).toBe(true);
  }
  expect(html).toContain('fictional-reference');
  expect(html).toContain('fictional-candidate');
  expect(html).toContain('fictional-old');
  expect(html).toContain('fictional-new');
});
