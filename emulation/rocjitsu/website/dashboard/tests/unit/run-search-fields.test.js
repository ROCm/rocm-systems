import { expect, test, vi } from 'vitest';
import { runSearchFields, runSelectionLabelWithBranch } from '../../src/components/shared/runOptionPresentation.js';
import { formatRunTime } from '../../src/utils/formatters.js';
import { filterRunOptions } from '../../src/components/branch/branchPresentation.js';
import RunSelector from '../../src/components/compare/RunSelector.jsx';

const run = { runId: 'attempt-needle', branch: 'topic/branch-needle', targets: ['gfx1250'], modes: ['default'],
  timestamp: '2026-10-05T14:00:00Z', commitTimestamp: '2026-10-04T10:00:00Z',
  provenance: { rocjitsuCommitSha: 'abcdef1'.padEnd(40, 'a'), commitMessage: 'message-needle' }, pullRequest: { number: 314159 } };

test('shared search fields include run and source identity plus raw and displayed clocks', () => {
  expect(runSearchFields(run)).toEqual([
    runSelectionLabelWithBranch(run), `Test run ${formatRunTime(run.timestamp)}`, `Commit ${formatRunTime(run.commitTimestamp)}`,
    run.runId, run.branch, 'gfx1250', run.timestamp, run.commitTimestamp,
    run.provenance.rocjitsuCommitSha, 'message-needle',
  ]);
});

function compareFilter(options, query) {
  const selector = RunSelector({ label: 'Candidate run', options, value: null, onChange() {} });
  return selector.props.children.props.filterOptions(options, { inputValue: query });
}

test.each(['attempt-needle', 'BRANCH-NEEDLE', 'message-needle', 'abcdef1', 'gfx1250', '2026-10-04', '2026-10-05'])('both pickers search common field %s', (query) => {
  expect(filterRunOptions([run], { inputValue: query })).toEqual([run]);
  expect(compareFilter([run], query)).toEqual([run]);
});

test('both pickers use locale-independent identity search under a Turkish default locale', () => {
  const localeLowerCase = String.prototype.toLocaleLowerCase;
  // Model the browser default locale while retaining native Turkish case folding.
  const locale = vi.spyOn(String.prototype, 'toLocaleLowerCase').mockImplementation(function (locales) {
    return localeLowerCase.call(this, locales ?? 'tr');
  });
  try {
    const options = [{ ...run, branch: 'ISSUE' }];
    for (const query of ['ISSUE', 'issue', 'ıssue']) {
      const expected = query === 'ıssue' ? [] : options;
      expect(filterRunOptions(options, { inputValue: query })).toEqual(expected);
      expect(compareFilter(options, query)).toEqual(expected);
    }
  } finally {
    locale.mockRestore();
  }
});

test('branch retains PR search while Compare retains its disclosed 50-match cap', () => {
  expect(filterRunOptions([run], { inputValue: '314159' })).toEqual([run]);
  expect(compareFilter([run], '314159')).toEqual([]);
  const options = Array.from({ length: 60 }, (_, index) => ({ ...run, runId: `attempt-${index}` }));
  expect(filterRunOptions(options, { inputValue: '' })).toHaveLength(60);
  expect(compareFilter(options, '')).toEqual(options.slice(0, 50));
});

test.each(['default', 'single'])('neither picker searches thread-policy mode %s: a run holds both, so it says nothing about the run', (query) => {
  expect(filterRunOptions([run], { inputValue: query })).toEqual([]);
  expect(compareFilter([run], query)).toEqual([]);
});

test('the compare picker finds a run by pasting the text it shows, branch included', () => {
  const shown = runSelectionLabelWithBranch(run);
  expect(shown).toContain(run.branch);
  expect(compareFilter([run], shown)).toEqual([run]);
  expect(filterRunOptions([run], { inputValue: shown })).toEqual([run]);
});
