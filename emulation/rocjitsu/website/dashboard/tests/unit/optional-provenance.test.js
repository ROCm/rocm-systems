import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { readMaintainedExamplePublication } from '../helpers/maintainedExamplePublication.js';
import BranchEnvironment from '../../src/components/branch/BranchEnvironment.jsx';
import RunMetadataDiff from '../../src/components/compare/RunMetadataDiff.jsx';

test('optional provenance renders em dashes with zero/false intact and source nulls unchanged', () => {
  const source = readMaintainedExamplePublication();
  source.runs[0].provenance = { rocjitsu: { rocjitsu_commit_sha: null, rocjitsu_commit_timestamp: null }, auxiliary: { zero: 0, enabled: false } };
  source.runs[0].publication.source.pull_request = { number: 12 };
  const before = structuredClone(source);
  const { data, sourceData } = adaptPublishedDashboardData(source);
  const run = data.runs[0];
  const branch = renderToStaticMarkup(createElement(BranchEnvironment, { candidate: run, selection: { target: 'gfx950', mode: 'default' } }));
  expect(branch).toMatch(/branch-source-commit[\s\S]*?>—<\/td>/);
  expect(branch).toContain('Benchmark set');
  const metadata = renderToStaticMarkup(createElement(RunMetadataDiff, { baseline: run, candidate: run, filters: { targets: run.targets, suites: data.suites, modes: run.modes } }));
  expect(metadata).toContain('>0</td>');
  expect(metadata).toContain('>false</td>');
  expect(run.pullRequest).toEqual({ number: 12 });
  expect(run.sourceBase).toBeUndefined();
  expect(sourceData).toEqual(before);
});
