import { expect, test } from 'vitest';
import * as presentation from '../../src/components/branch/branchPresentation.js';

const attempt = { runId: 'fictional-exact-attempt-72', branch: 'fictional/searchable-branch', timestamp: '2026-10-05T14:00:00Z',
  pullRequest: { number: 42 }, provenance: { rocjitsuCommitSha: 'b'.repeat(40), commitMessage: 'Fictional dispatch batch' } };

test('attempt selector labels and search identify branch, full SHA, PR, message, timestamp, and exact attempt without truncating published options', () => {
  expect(presentation.runOptionLabel?.(attempt)?.includes(attempt.runId)).toBe(true);
  const options = Array.from({ length: 75 }, (_, index) => ({ ...attempt, runId: `fictional-attempt-${index}` }));
  expect(presentation.filterRunOptions(options, { inputValue: '' })).toHaveLength(75);
  for (const inputValue of [attempt.branch, 'b'.repeat(40), '42', 'dispatch batch', '2026-10-05T14:00:00Z', attempt.runId]) {
    expect(presentation.filterRunOptions([attempt], { inputValue })).toEqual([attempt]);
  }
  expect(presentation.filterRunOptions([attempt], { inputValue: 'not-published' })).toEqual([]);
});

test.each(['09:30', '14:00', '2026-10-02T09:30:00Z', 'single', 'gfx950'])('branch attempt search includes displayed context %s', (inputValue) => {
  const run = { ...attempt, commitTimestamp: '2026-10-02T09:30:00Z', modes: ['single'], targets: ['gfx950'] };
  expect(presentation.filterRunOptions([run], { inputValue })).toEqual([run]);
});

test('configuration availability uses explicit published modes and positive matched coverage, while measured zero is not an absent result', () => {
  const run = { configurations: [{ target: 'gfx1250', thread_policy: 'single' }] };
  const noPairs = { comparable: [], notComparable: [], baselineDuration: 0, candidateDuration: 0, aggregateDelta: null };
  expect(presentation.configurationState?.(run, run, noPairs, 'gfx1250', 'single')).toMatchObject({ available: false, deltaSeconds: null, deltaPercent: null });
  expect(presentation.configurationState(run, run, noPairs, 'gfx950', 'default').reason).toContain('Candidate configuration not published');
  expect(presentation.configurationState(run, null, noPairs, 'gfx1250', 'single').reason).toContain('No reference selected');
  const measured = { ...noPairs, comparable: [{ candidateTest: { durationSeconds: 0 }, baselineTest: { durationSeconds: 0 } }] };
  expect(presentation.configurationState(run, run, measured, 'gfx1250', 'single')).toMatchObject({ available: true, matched: 1, deltaSeconds: 0, deltaPercent: null });
});

test('configuration reasons preserve missing-run precedence and exact optional presence checks without mutation', () => {
  const run = { configurations: [{ target: 'gfx1250', thread_policy: 'single' }] };
  const comparison = { comparable: [{}], notComparable: [], candidateDuration: 0, baselineDuration: 0, aggregateDelta: null };
  const absent = 'Candidate configuration not published';
  const cases = [
    { candidate: null, baseline: null, reason: 'Candidate attempt unavailable' },
    { candidate: {}, baseline: null, reason: 'No reference selected' },
    ...[undefined, null, [], [{ target: 'gfx950', thread_policy: 'single' }], [{ target: 'gfx1250', thread_policy: 'default' }]].map((configurations) => ({ candidate: { configurations }, baseline: run, reason: absent })),
    { candidate: run, baseline: {}, reason: 'Reference configuration not published' },
    { candidate: run, baseline: run, reason: null },
    { candidate: run, baseline: run, target: null, reason: absent },
    { candidate: run, baseline: run, target: undefined, reason: absent },
    { candidate: run, baseline: run, mode: null, reason: absent },
    { candidate: run, baseline: run, mode: undefined, reason: absent },
    { candidate: { configurations: [{ target: 1250, thread_policy: 'single' }] }, baseline: run, target: '1250', reason: absent },
    { candidate: { configurations: [{ target: 'gfx1250', thread_policy: 1 }] }, baseline: run, mode: '1', reason: absent },
  ];
  const before = structuredClone(cases);
  for (const entry of cases) {
    const target = Object.hasOwn(entry, 'target') ? entry.target : 'gfx1250';
    const mode = Object.hasOwn(entry, 'mode') ? entry.mode : 'single';
    expect(presentation.configurationState(entry.candidate, entry.baseline, comparison, target, mode)).toEqual({
      matched: 1, excluded: 0, available: entry.reason === null, reason: entry.reason,
      deltaSeconds: entry.reason ? null : 0, deltaPercent: null,
    });
  }
  expect(cases).toEqual(before);
  for (const [notComparable, reason] of [[[], 'No benchmarks in the selected suites or search'], [[{}], 'No matched completed benchmarks']]) {
    expect(presentation.configurationState(run, run, { ...comparison, comparable: [], notComparable }, 'gfx1250', 'single').reason).toBe(reason);
  }
});

test('real suite grouping sorts signed measured seconds by default and signed percentages on request', () => {
  const pair = (name, suite, base, next) => ({ candidateTest: { name, suite, testId: name, durationSeconds: next }, baselineTest: { durationSeconds: base }, delta: base ? ((next - base) / base) * 100 : null });
  const rows = [pair('small', 'Suite A', 1, 2), pair('large', 'Suite A', 100, 80), pair('zero', 'Suite A', 0, 0), pair('other', 'Suite B', 60, 55)];
  expect(presentation.groupComparisonRows?.(rows).map((group) => [group.suite, group.rows.map((row) => row.candidateTest.name)])).toEqual([['Suite A', ['small', 'zero', 'large']], ['Suite B', ['other']]]);
  expect(presentation.groupComparisonRows(rows, 'percent').flatMap((group) => group.rows.map((row) => row.candidateTest.name))).toEqual(['small', 'large', 'zero', 'other']);
});

test('source links accept safe absolute HTTP(S) only, without credentials or unsafe protocols', () => {
  expect(presentation.safeExternalUrl?.('https://github.com/ROCm/rocm-systems/actions/runs/123')).toBe('https://github.com/ROCm/rocm-systems/actions/runs/123');
  for (const url of ['javascript:alert(1)', 'data:text/html,test', '//example.com/path', '/local/path', 'https://user:password@example.com/path']) expect(presentation.safeExternalUrl(url)).toBeNull();
  expect(presentation.safeExternalUrl(undefined)).toBeNull();
});
