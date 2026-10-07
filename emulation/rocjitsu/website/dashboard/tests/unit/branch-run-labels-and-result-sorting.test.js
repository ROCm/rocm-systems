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

test('configuration availability uses explicit published modes and positive matched coverage, while measured zero is not an absent result', () => {
  const run = { configurations: [{ target: 'gfx1250', mode: 'ST' }] };
  const noPairs = { comparable: [], notComparable: [], baselineDuration: 0, candidateDuration: 0, aggregateDelta: null };
  expect(presentation.configurationState?.(run, run, noPairs, 'gfx1250', 'ST')).toMatchObject({ available: false, deltaSeconds: null, deltaPercent: null });
  expect(presentation.configurationState(run, run, noPairs, 'gfx950', 'MT').reason).toContain('Candidate configuration not published');
  expect(presentation.configurationState(run, null, noPairs, 'gfx1250', 'ST').reason).toContain('No reference selected');
  const measured = { ...noPairs, comparable: [{ candidateTest: { durationSeconds: 0 }, baselineTest: { durationSeconds: 0 } }] };
  expect(presentation.configurationState(run, run, measured, 'gfx1250', 'ST')).toMatchObject({ available: true, matched: 1, deltaSeconds: 0, deltaPercent: null });
});

test('real suite grouping sorts signed measured seconds by default and signed percentages on request', () => {
  const pair = (name, suite, base, next) => ({ candidateTest: { name, suite, testId: name, durationSeconds: next }, baselineTest: { durationSeconds: base }, delta: base ? ((next - base) / base) * 100 : null });
  const rows = [pair('small', 'Suite A', 1, 2), pair('large', 'Suite A', 100, 80), pair('zero', 'Suite A', 0, 0), pair('other', 'Suite B', 60, 55)];
  expect(presentation.groupComparisonRows?.(rows).map((group) => [group.suite, group.rows.map((row) => row.candidateTest.name)])).toEqual([['Suite A', ['small', 'zero']], ['Suite B', ['other']], ['Suite A', ['large']]]);
  expect(presentation.groupComparisonRows(rows, 'percent').flatMap((group) => group.rows.map((row) => row.candidateTest.name))).toEqual(['small', 'other', 'large', 'zero']);
});

test('source links accept safe absolute HTTP(S) only, without credentials or unsafe protocols', () => {
  expect(presentation.safeExternalUrl?.('https://github.com/ROCm/rocm-systems/actions/runs/123')).toBe('https://github.com/ROCm/rocm-systems/actions/runs/123');
  for (const url of ['javascript:alert(1)', 'data:text/html,test', '//example.com/path', '/local/path', 'https://user:password@example.com/path']) expect(presentation.safeExternalUrl(url)).toBeNull();
  expect(presentation.safeExternalUrl(undefined)).toBeNull();
});
