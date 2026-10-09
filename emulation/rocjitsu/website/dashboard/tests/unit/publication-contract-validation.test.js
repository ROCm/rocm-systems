import { expect, test } from 'vitest';
import { loadDashboardData, validatePublishedDashboardData, validatePublishedResult } from '../../src/data/dashboardValidation.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

const invalidCases = [
  ['unsafe repository', (s) => { s.siteConfig = { repository: 'javascript:alert(1)', isBeta: true, canonicalBranch: 'develop' }; }, /repository URL/],
  ['credential URL', (s) => { s.siteConfig = { repository: 'https://name:secret@example.test/', isBeta: true, canonicalBranch: 'develop' }; }, /repository URL/],
  ['noncanonical site branch', (s) => { s.siteConfig = { repository: 'https://example.test/repo', isBeta: true, canonicalBranch: 'main' }; }, /canonicalBranch/],
  ['impossible calendar day', (s) => { s.index.generatedAt = '2026-02-30T12:00:00Z'; }, /ISO/],
  ['unsafe run path', (s) => { s.index.runFiles[0] = 'runs/../../escape.json'; }, /filename/],
  ['duplicate file', (s) => { s.index.runFiles[1] = s.index.runFiles[0]; }, /Duplicate/],
  ['filename identity mismatch', (s) => { s.runs[0].id = 'different'; }, /filename/],
  ['unsafe catalog path', (s) => { s.runs[0].testCatalog = '../escape.json'; }, /catalog/],
  ['catalog identity mismatch', (s) => { s.catalogs[s.runs[0].testCatalog].id = 'different'; }, /catalog contract/],
  ['invalid configuration key', (s) => { s.catalogs[s.runs[0].testCatalog].configurations['gfx1250:unknown'] = ['a']; }, /invalid configuration/],
  ['duplicate workload definition', (s) => { const c = s.catalogs[s.runs[0].testCatalog]; c.tests.push(c.tests[0]); }, /duplicate test definition/],
  ['omitted workload', (s) => { s.runs[0].configurations[0].results.pop(); }, /exactly one result/],
  ['duplicate result', (s) => { const r = s.runs[0].configurations[0].results; r.push(r[0]); }, /exactly one result/],
  ['unknown workload', (s) => { s.runs[0].configurations[0].results[0].testId = 'unknown'; }, /exactly one result/],
  ['duplicate configuration', (s) => { s.runs[0].configurations.push(s.runs[0].configurations[0]); }, /duplicate configuration/],
  ['missing explicit mode', (s) => { delete s.runs[0].configurations[0].threadingMode; }, /configuration/],
  ['negative duration', (s) => { s.runs[0].configurations[0].results[0].durationSeconds = -1; }, /nonnegative/],
  ['completed error', (s) => { s.runs[0].configurations[0].results[0].error = 'invalid'; }, /cannot contain an error/],
  ['failure duration', (s) => { s.runs[0].configurations[0].results[0].status = 'failed'; }, /null durationSeconds/],
  ['future execution', (s) => { s.runs[0].execution.completedAt = '2026-10-06T00:00:00Z'; }, /publication/],
  ['commit after completion', (s) => { s.runs[0].source.committedAt = '2026-10-06T00:00:00Z'; }, /committed/],
  ['invalid base SHA', (s) => { s.runs[0].source.base = { branch: 'develop', commit: 'abcd' }; }, /source base/],
  ['unsafe PR URL', (s) => { s.runs[0].source.pullRequest = { number: 1, url: 'https://evil.test/pull/1' }; }, /pullRequest/],
  ['duplicate environment key', (s) => { s.runs[0].environment.push(s.runs[0].environment[0]); }, /run contract/],
  ['changed reused workload', (s) => { s.catalogs['test-catalogs/fictional-current.json'].tests[0].problem.m = 999; }, /defined differently/],
  ['conflicting commit time', (s) => { s.runs[1].source.commit = s.runs[0].source.commit; }, /conflicting committedAt/],
];

test.each(invalidCases)('schema 2 rejects %s', (_, mutate, expected) => {
  const source = createSchema2Publication(); mutate(source);
  expect(() => validatePublishedDashboardData(source)).toThrow(expected);
});

test('independent branch machines and scalar environments remain disclosed', () => {
  const source = createSchema2Publication(); source.runs[0].environment = [];
  const { data } = validatePublishedDashboardData(source);
  expect(data.allRuns.at(-1).machineId).toBe('fictional-branch-node');
  expect(data.allRuns.at(-1).provenance.details).toEqual(source.runs[43].environment);
  expect(data.runs.some((run) => data.backfillRunIds.has(run.runId))).toBe(true);
  expect(loadDashboardData(structuredClone(data)).allRuns).toHaveLength(44);
});

test('result status/error distinction preserves zero and failure/timeout diagnostics', () => {
  expect(validatePublishedResult({ testId: 'zero', status: 'completed', durationSeconds: 0, error: null }).durationSeconds).toBe(0);
  for (const status of ['failed', 'timeout']) expect(validatePublishedResult({ testId: status, status, durationSeconds: null, error: 'diagnostic' }).status).toBe(status);
  expect(() => validatePublishedResult({ testId: 'missing', status: 'completed', error: null })).toThrow(/durationSeconds/);
  expect(() => validatePublishedResult({ testId: 'bad', status: 'failed', durationSeconds: null, error: '' })).toThrow(/non-empty/);
  expect(() => validatePublishedResult({ testId: 'bad', status: 'completed', durationSeconds: Infinity, error: null })).toThrow(/finite/);
});
