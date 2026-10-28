import { expect, test } from 'vitest';
import RunOptionDetails from '../../src/components/shared/RunOptionDetails.jsx';
import RunSelector from '../../src/components/compare/RunSelector.jsx';
import CommitComparison from '../../src/components/shared/CommitComparison.jsx';
import { monoFont } from '../../src/theme/tokens.js';

const run = { runId: 'attempt', branch: 'develop', modes: ['single'], provenance: { rocjitsuCommitSha: 'a'.repeat(40) } };

test('run option SHA uses the bundled identity font stack', () => {
  const details = RunOptionDetails({ run });
  expect(details.props.secondary.props.children[0].props.sx.fontFamily).toBe(monoFont);
});

test('selected comparison attempt uses the bundled identity font stack', () => {
  const selector = RunSelector({ label: 'Candidate run', options: [run], value: run, onChange() {} });
  expect(selector.props.children[1].props.sx.fontFamily).toBe(monoFont);
});

test('both comparison SHAs use the bundled identity font stack', () => {
  const comparison = CommitComparison({ candidate: run, baseline: run });
  for (const index of [0, 2]) expect(comparison.props.children[index].props.sx.fontFamily).toBe(monoFont);
});
