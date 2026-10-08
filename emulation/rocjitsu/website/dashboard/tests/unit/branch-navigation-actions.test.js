import { isValidElement } from 'react';
import { expect, test } from 'vitest';
import BenchmarkDifferences from '../../src/components/branch/BenchmarkDifferences.jsx';
import BranchList from '../../src/components/branch/BranchList.jsx';
import ConfigurationMatrix from '../../src/components/branch/ConfigurationMatrix.jsx';
import { selectConfigurationComparison } from '../../src/data/branchSelectors.js';
import { configurationState } from '../../src/components/branch/branchPresentation.js';

function nodes(element) {
  if (Array.isArray(element)) return element.flatMap(nodes);
  if (!isValidElement(element)) return [];
  return [element, ...nodes(element.props.children)];
}
const candidate = { runId: 'fictional-candidate-exact', configurations: [{ target: 'gfx950', threadingMode: 'MT' }], tests: [] };
const reference = { ...candidate, runId: 'fictional-manual-reference-exact' };
const selection = { branch: 'fictional/branch', candidateId: candidate.runId, referenceId: reference.runId, manual: true, target: 'gfx950', mode: 'MT', detail: true };

const tree = (suites, baseline, onOpenComparison) => {
  const comparison = selectConfigurationComparison(candidate, baseline, { target: 'gfx950', mode: 'MT', suites });
  return BenchmarkDifferences({ comparison, summary: configurationState(candidate, baseline, comparison, 'gfx950', 'MT'), candidate, reference: baseline, selection,
    suites, suiteOptions: ['Fictional suite'], query: '', sort: 'seconds', onOpenComparison,
  });
};

test('full comparison action forwards the exact pair and configuration with explicit local suite scope, even when scope is empty', () => {
  for (const suites of [['Fictional suite'], []]) {
    const calls = [];
    const action = nodes(tree(suites, reference, (payload) => calls.push(payload))).find((node) => node.props['data-testid'] === 'branch-open-comparison');
    expect(action?.props.disabled).toBe(false);
    action.props.onClick();
    expect(calls).toEqual([{ candidateId: candidate.runId, baselineId: reference.runId, target: 'gfx950', mode: 'MT', suites }]);
  }
});

test('full comparison action is visibly disabled when the exact reference is unavailable', () => {
  const action = nodes(tree(['Fictional suite'], null, () => { throw new Error('Unavailable pair must not open'); })).find((node) => node.props['data-testid'] === 'branch-open-comparison');
  expect(action?.props.disabled).toBe(true);
});

test('branch list exposes its scroll position and selects exact entry through keyboard-accessible controls', () => {
  const entry = { branch: 'fictional/branch', latestRun: { ...candidate, timestamp: '2026-10-05T14:00:00Z', provenance: { rocjitsuCommitSha: 'b'.repeat(40) } } };
  const selections = []; const positions = [];
  const list = BranchList({ branches: [entry], total: 1, selectedBranch: entry.branch, onSelect: (branch) => selections.push(branch), onScroll: (event) => positions.push(event.currentTarget.scrollTop) });
  const region = nodes(list).find((node) => node.props['data-testid'] === 'branch-list-scroll');
  expect(typeof region.props.onScroll).toBe('function');
  region.props.onScroll({ currentTarget: { scrollTop: 333 } });
  expect(positions).toEqual([333]);
  const row = nodes(list).find((node) => node.props['data-testid'] === `branch-row-${entry.branch}`);
  expect(row.props['aria-pressed']).toBe(true);
  row.props.onClick();
  expect(selections).toEqual([entry]);
});

test('all configuration cells including unpublished ones remain inspectable and emit an exact target/mode pair', () => {
  const scope = [];
  const matrix = ConfigurationMatrix({ candidate, reference, selection, suites: [], onSelect: (target, mode) => scope.push([target, mode]) });
  const cells = nodes(matrix).filter((node) => node.props['data-testid']?.startsWith('branch-config-'));
  expect(cells).toHaveLength(4);
  for (const cell of cells) { expect(cell.props.disabled).not.toBe(true); cell.props.onClick(); }
  expect(scope).toEqual([['gfx1250', 'ST'], ['gfx950', 'ST'], ['gfx1250', 'MT'], ['gfx950', 'MT']]);
});
