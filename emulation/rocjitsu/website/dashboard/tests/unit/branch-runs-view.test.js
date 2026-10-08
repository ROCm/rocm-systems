import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import BranchRunsView from '../../src/components/views/BranchRunsView.jsx';
import { selectConfigurationComparison } from '../../src/data/branchSelectors.js';
import { formatDuration } from '../../src/utils/formatters.js';

// Fictional normalized schema-2 inputs, confined to this unit test.
const sha = (digit) => digit.repeat(40);
const testResult = (id, durationSeconds, target = 'gfx1250', mode = 'ST', suite = 'Fictional suite') => ({
  testId: `${target}:${mode}:${id}`, logicalTestId: id, name: `Fictional ${id}`, suite, problem: {},
  target, mode, durationSeconds, status: 'completed', error: null,
});
const run = (runId, branch, timestamp, digit, tests = [testResult('work', 10)]) => ({
  runId, branch, timestamp, commitTimestamp: timestamp,
  catalogId: 'fictional-catalog',
  configurations: [...new Map(tests.map((test) => [`${test.target}:${test.mode}`, { target: test.target, threadingMode: test.mode }])).values()],
  targets: [...new Set(tests.map((test) => test.target))], modes: [...new Set(tests.map((test) => test.mode))],
  provenance: { rocjitsuCommitSha: sha(digit), commitMessage: `Fictional ${runId}`, details: [] },
  tests, machineId: 'fictional-machine', trigger: 'manual',
});
const baseline = run('fictional-develop-attempt', 'develop', '2026-10-04T15:00:00Z', 'a');
const candidate = { ...run('fictional-branch-attempt-2', 'users/fictional/newest', '2026-10-05T14:00:00Z', 'b'),
  sourceBase: { branch: 'develop', commit: sha('a') }, pullRequest: { number: 17 },
};
const older = run('fictional-branch-attempt-1', candidate.branch, '2026-09-01T14:00:00Z', 'c');
const quiet = run('fictional-inactive', 'users/fictional/old', '2026-09-01T14:00:00Z', 'd');
const second = run('fictional-second', 'users/fictional/second', '2026-10-03T14:00:00Z', 'e');
const data = { schemaVersion: 2, generatedAt: '2026-10-05T15:00:00Z', canonicalBranch: 'develop',
  allRuns: [quiet, second, older, baseline, candidate], runs: [baseline], suites: ['Fictional suite'],
};

const render = (data, selection = {}) => renderToStaticMarkup(createElement(BranchRunsView, {
  data,
  state: { branchSelection: selection, setBranchSelection() { throw new Error('Rendering defaults must not write route history'); } },
  onOpenComparison() {},
}));

test('no published branch attempts renders an honest empty state instead of preview measurements', () => {
  const html = render({ generatedAt: '2026-10-05T15:00:00Z', canonicalBranch: 'develop', allRuns: [] });
  expect(html).toContain('No published branch runs');
  expect(html).not.toContain('Frontend preview');
  expect(html).not.toContain('example-candidate');
});

test('published branch rows use the activity window and newest execution ordering, never the canonical-only set', () => {
  const html = render(data);
  expect(html.includes('data-testid="branch-row-users/fictional/newest"')).toBe(true);
  expect(html.includes('data-testid="branch-row-users/fictional/second"')).toBe(true);
  expect(html.includes('data-testid="branch-row-users/fictional/old"')).toBe(false);
  expect(html.indexOf('branch-row-users/fictional/newest')).toBeLessThan(html.indexOf('branch-row-users/fictional/second'));
  expect(html.includes('Search branch, PR or SHA')).toBe(true);
  expect(html.includes('With PR')).toBe(true);
  expect(html.includes('No PR')).toBe(true);
  expect(html.includes('Newest published executions first')).toBe(true);
  expect(html.includes('data-screen="list"')).toBe(true);
});

test('default pair exposes exact candidate SHA and attempt with automatic develop-base reason and no manual reset action', () => {
  const html = render(data);
  expect(html.includes('data-testid="candidate-selected-identity"')).toBe(true);
  expect(html.includes(sha('b'))).toBe(true);
  expect(html.includes(candidate.runId)).toBe(true);
  expect(html.includes('data-testid="reference-selected-identity"')).toBe(true);
  expect(html.includes(sha('a'))).toBe(true);
  expect(html.includes(baseline.runId)).toBe(true);
  expect(html.includes('Exact develop base')).toBe(true);
  expect(html.includes('Restore automatic base')).toBe(false);
  expect(/<h2[^>]*id="branch-detail-heading"[^>]*tabindex="-1"/.test(html)).toBe(true);
});

test('four matrix cells expose explicit availability and selected scope rather than invented missing-mode deltas', () => {
  const html = render(data, { branch: candidate.branch, candidateId: candidate.runId, referenceId: baseline.runId, target: 'gfx950', mode: 'MT', detail: true });
  for (const target of ['gfx1250', 'gfx950']) for (const mode of ['ST', 'MT']) {
    expect(html.includes(`data-testid="branch-config-${target}-${mode}"`)).toBe(true);
  }
  expect(html.includes('data-testid="branch-config-gfx950-MT" aria-pressed="true"')).toBe(true);
  expect(html.includes('Candidate configuration not published')).toBe(true);
  expect(html.includes('data-testid="branch-selected-configuration"')).toBe(true);
  expect(html.includes('0 matched · 0 excluded')).toBe(true);
  expect(html.includes('data-comparison-state="unavailable"')).toBe(true);
  expect(html.includes('Back to branches')).toBe(true);
});

test('difference table and stacked phone rows share actual matched sums, suite grouping, and symmetric exclusions', () => {
  const base = { ...baseline, tests: [testResult('small', 1), testResult('large', 100), testResult('timeout', 10), testResult('base-only', 15)] };
  const next = { ...candidate, tests: [testResult('small', 2), testResult('large', 80), { ...testResult('timeout', null), status: 'timeout', error: 'Fictional timeout diagnostic' }, testResult('new-only', 20)] };
  const source = { ...data, allRuns: [base, next], runs: [base] };
  const model = selectConfigurationComparison(next, base, { target: 'gfx1250', mode: 'ST', suites: ['Fictional suite'] });
  const html = render(source);
  expect(model.comparable).toHaveLength(2);
  expect(model.notComparable).toHaveLength(3);
  expect(html.includes('aria-label="Benchmark differences"')).toBe(true);
  expect(html.includes('data-testid="branch-mobile-results"')).toBe(true);
  expect(html.includes('2 matched · 3 excluded')).toBe(true);
  expect(html.includes(formatDuration(model.baselineDuration))).toBe(true);
  expect(html.includes(formatDuration(model.candidateDuration))).toBe(true);
  expect(html.includes('Search benchmarks')).toBe(true);
  expect(html.includes('Abs change')).toBe(true);
  expect(html.includes('Pct change')).toBe(true);
  expect(html.indexOf('branch-result-gfx1250:ST:small')).toBeLessThan(html.indexOf('branch-result-gfx1250:ST:large'));
  expect(html.includes('Excluded results (3)')).toBe(true);
  expect(html.includes('Timeout')).toBe(true);
  expect(html.includes('Fictional base-only')).toBe(true);
  expect(html.includes('Fictional new-only')).toBe(true);
});

test('environment disclosure aligns arbitrary facts including zero and false and renders only safe provided source links', () => {
  const base = { ...baseline, pullRequest: { number: 1, url: 'javascript:alert(1)' }, provenance: { ...baseline.provenance, details: [{ key: 'baseline-only', label: 'Fictional baseline tool', value: 'old-tool' }, { key: 'flag', label: 'Fictional flag', value: false }] } };
  const next = { ...candidate, pullRequest: { number: 17, url: 'https://github.com/ROCm/rocm-systems/pull/17' }, provenance: { ...candidate.provenance, details: [{ key: 'candidate-only', label: 'Fictional candidate tool', value: '<script>not code</script>' }, { key: 'flag', label: 'Fictional flag', value: 0 }] } };
  const html = render({ ...data, repository: 'https://github.com/ROCm/rocm-systems', allRuns: [base, next], runs: [base] });
  expect(html.includes('Environment &amp; measurement details')).toBe(true);
  expect(html.includes('data-testid="branch-environment-baseline-only"')).toBe(true);
  expect(html.includes('data-testid="branch-environment-candidate-only"')).toBe(true);
  expect(html.includes('old-tool')).toBe(true);
  expect(html.includes('&lt;script&gt;not code&lt;/script&gt;')).toBe(true);
  expect(html.includes('>false<')).toBe(true);
  expect(html.includes('>0<')).toBe(true);
  expect(html).not.toContain('Workflow');
  expect(html.includes('href="https://github.com/ROCm/rocm-systems/pull/17"')).toBe(true);
  expect(html.includes('href="javascript:')).toBe(false);
});

test('unavailable exact identities stay inspectable and cannot silently fall back to a different published pair', () => {
  const html = render(data, { branch: candidate.branch, candidateId: 'fictional-missing-candidate', referenceId: 'fictional-missing-reference', manual: false, detail: true });
  expect(html.includes('Unavailable attempt: fictional-missing-candidate')).toBe(true);
  expect(html.includes('Unavailable attempt: fictional-missing-reference')).toBe(true);
  expect(html.includes('Selected candidate is not in the current published data')).toBe(true);
  expect(html.includes('Saved reference attempt is unavailable')).toBe(true);
  expect(html.includes('aria-label="Benchmark differences"')).toBe(false);
  expect(html.includes('Restore automatic base')).toBe(false);
  const manual = render(data, { branch: candidate.branch, candidateId: candidate.runId, referenceId: second.runId, manual: true, detail: true });
  expect(manual.includes('Manual reference')).toBe(true);
  expect(manual.includes('Restore automatic base')).toBe(true);
});

test('pending published data has a scoped loading state and never stale demonstration results', () => {
  const html = render(null);
  expect(html.includes('Loading published branch runs')).toBe(true);
  expect(html.includes('role="status"')).toBe(true);
  expect(html.includes('example-')).toBe(false);
});

test('selected target/mode rows and matched totals change together for real published configurations', () => {
  const base = { ...baseline, configurations: [{ target: 'gfx1250', threadingMode: 'ST' }, { target: 'gfx950', threadingMode: 'MT' }], tests: [testResult('st-work', 10), testResult('mt-work', 60, 'gfx950', 'MT')] };
  const next = { ...candidate, configurations: base.configurations, tests: [testResult('st-work', 8), testResult('mt-work', 72, 'gfx950', 'MT')] };
  for (const [target, mode, name, other] of [['gfx1250', 'ST', 'st-work', 'mt-work'], ['gfx950', 'MT', 'mt-work', 'st-work']]) {
    const model = selectConfigurationComparison(next, base, { target, mode, suites: ['Fictional suite'] });
    const html = render({ ...data, allRuns: [base, next], runs: [base] }, { branch: next.branch, candidateId: next.runId, referenceId: base.runId, target, mode });
    expect(html.includes(`data-testid="branch-result-${target}:${mode}:${name}"`)).toBe(true);
    expect(html.includes(`branch-result-gfx1250:ST:${other}`)).toBe(false);
    expect(html.includes(`branch-result-gfx950:MT:${other}`)).toBe(false);
    expect(html.includes(formatDuration(model.baselineDuration))).toBe(true);
    expect(html.includes(formatDuration(model.candidateDuration))).toBe(true);
    expect(html.includes('1 matched · 0 excluded')).toBe(true);
  }
});

test('measured zero remains a matched result but never manufactures a percentage against a zero reference', () => {
  const base = { ...baseline, tests: [testResult('measured-zero', 0)] };
  const next = { ...candidate, tests: [testResult('measured-zero', 0)] };
  const html = render({ ...data, allRuns: [base, next], runs: [base] });
  expect(html.includes('data-testid="branch-result-gfx1250:ST:measured-zero"')).toBe(true);
  expect(html.includes('1 matched · 0 excluded')).toBe(true);
  expect(html.includes('Percentage unavailable: zero reference runtime')).toBe(true);
  expect(html.includes('NaN')).toBe(false);
  expect(html.includes('Infinity')).toBe(false);
});
