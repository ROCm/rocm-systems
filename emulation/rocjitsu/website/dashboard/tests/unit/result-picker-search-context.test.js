import { expect, test } from 'vitest';
import ChartPointSelector from '../../src/components/benchmarks/ChartPointSelector.jsx';
import { benchmarkResultChoices } from '../../src/components/benchmarks/benchmarkExplorer.js';

const run = {
  runId: 'fictional-search', branch: 'develop', timestamp: '2026-10-04T14:00:00Z',
  commitTimestamp: '2026-10-02T09:30:00Z', provenance: { rocjitsuCommitSha: 'a'.repeat(40) },
};
const options = benchmarkResultChoices({ runs: [run], series: [{ name: 'gfx950 single', target: 'gfx950', mode: 'single', records: [{ test: { status: 'completed', durationSeconds: 2 } }] }] });

test.each(['develop', '14:00', '09:30', 'fictional-search', 'gfx950 single'])('result search includes displayed context: %s', (query) => {
  const selector = ChartPointSelector({ options, selectedId: options[0].id });
  const autocomplete = selector.props.children[1];
  expect(autocomplete.props.filterOptions(options, { inputValue: query, getOptionLabel: autocomplete.props.getOptionLabel })).toEqual(options);
});
