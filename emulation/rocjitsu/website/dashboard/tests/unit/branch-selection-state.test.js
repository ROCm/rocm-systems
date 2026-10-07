import { expect, test } from 'vitest';
import * as selection from '../../src/components/branch/branchSelection.js';

const current = { branch: 'fictional/old', candidateId: 'fictional-old-attempt', referenceId: 'fictional-pinned', manual: true, target: 'gfx950', mode: 'MT', detail: false };

test('changing branch replaces a manual pair with latest candidate and automatic reference in one complete snapshot', () => {
  const next = selection.selectionForBranch(current, { branch: 'fictional/new', latestRun: { runId: 'fictional-new-attempt' } }, 'fictional-exact-base');
  expect(next).toEqual({ branch: 'fictional/new', candidateId: 'fictional-new-attempt', referenceId: 'fictional-exact-base', manual: false, target: 'gfx950', mode: 'MT', detail: true });
});

test('changing candidate recomputes automatic reference but retains an explicitly manual reference', () => {
  const automatic = selection.selectionForCandidate?.({ ...current, manual: false }, 'fictional-rerun', 'fictional-earlier-develop');
  expect(automatic).toEqual({ ...current, candidateId: 'fictional-rerun', referenceId: 'fictional-earlier-develop', manual: false });
  expect(selection.selectionForCandidate(current, 'fictional-rerun', 'fictional-earlier-develop')).toEqual({ ...current, candidateId: 'fictional-rerun' });
});

test('render-time defaults are complete without replacing exact missing URL identities', () => {
  const branches = [{ branch: 'fictional/new', latestRun: { runId: 'fictional-new-attempt' } }];
  const runs = [{ runId: 'fictional-new-attempt', branch: 'fictional/new', timestamp: '2026-10-05T14:00:00Z' }];
  expect(selection.resolveBranchSelection?.({}, { branches, runs, automaticReferenceId: 'fictional-base' })).toEqual({ branch: 'fictional/new', candidateId: 'fictional-new-attempt', referenceId: 'fictional-base', manual: false, target: 'gfx1250', mode: 'ST', detail: false });
  const missing = { ...current, candidateId: 'not-published', referenceId: 'missing-base', manual: false };
  expect(selection.resolveBranchSelection(missing, { branches, runs, automaticReferenceId: 'fictional-base' })).toEqual(missing);
});

test('manual reference, automatic restore, configuration, and phone Back all preserve a complete exact snapshot', () => {
  const manual = selection.selectionForAction?.(current, { type: 'reference', runId: 'fictional-any-published' });
  expect(manual).toEqual({ ...current, referenceId: 'fictional-any-published', manual: true });
  const automatic = selection.selectionForAction(manual, { type: 'automatic', runId: 'fictional-auto' });
  expect(automatic).toEqual({ ...manual, referenceId: 'fictional-auto', manual: false });
  const scope = selection.selectionForAction(automatic, { type: 'configuration', target: 'gfx1250', mode: 'ST' });
  expect(scope).toEqual({ ...automatic, target: 'gfx1250', mode: 'ST' });
  expect(selection.selectionForAction({ ...scope, detail: true }, { type: 'back' })).toEqual({ ...scope, detail: false });
});
