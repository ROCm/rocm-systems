import { Children, isValidElement } from 'react';
import { expect, test, vi, beforeEach } from 'vitest';
import { Dialog, DialogActions, Button } from '@mui/material';
import BenchmarksView from '../../src/components/views/BenchmarksView.jsx';
import ChartPointSelector from '../../src/components/benchmarks/ChartPointSelector.jsx';
import BenchmarkResultDialog from '../../src/components/benchmarks/BenchmarkResultDialog.jsx';
import BenchmarkHistoryChart from '../../src/components/benchmarks/BenchmarkHistoryChart.jsx';

// Exercise the component's real event handlers with persistent hook slots, without
// claiming DOM focus/geometry coverage from the node-only unit runner.
const hooks = vi.hoisted(() => ({ values: [], cursor: 0 }));
vi.mock('react', async (original) => ({ ...await original(),
  useMemo: (compute) => compute(),
  useState: (initial) => {
    const index = hooks.cursor++;
    if (!(index in hooks.values)) hooks.values[index] = typeof initial === 'function' ? initial() : initial;
    return [hooks.values[index], (next) => { hooks.values[index] = typeof next === 'function' ? next(hooks.values[index]) : next; }];
  },
}));
const nodes = (element) => {
  const result = [];
  const visit = (node) => { if (!isValidElement(node)) return; result.push(node); Children.forEach(node.props.children, visit); };
  visit(element);
  return result;
};
const workload = { id: 'llama', name: 'Llama long workload', suite: 'Llama', problem: {} };
const runs = ['first', 'second'].map((runId, index) => ({ runId, timestamp: `2026-10-0${index + 1}T12:00:00Z`, commitTimestamp: `2026-10-0${index + 1}T11:00:00Z`, provenance: { rocjitsuCommitSha: String(index).repeat(40) }, tests: [{ logicalTestId: 'llama', name: workload.name, target: 'gfx1250', mode: 'ST', suite: 'Llama', status: 'completed', durationSeconds: index + 1 }] }));
const props = { data: { testCatalog: [workload], runs }, filters: { targets: ['gfx1250'], modes: ['ST'], suites: ['Llama'] } };
const render = () => { hooks.cursor = 0; return BenchmarksView(props); };
const inspector = (tree) => nodes(tree).find((node) => node.type === Dialog && node.props['aria-labelledby'] === 'inspect-results-title');
const selector = (tree) => nodes(tree).find((node) => node.type === ChartPointSelector);
const details = (tree) => nodes(tree).find((node) => node.type === BenchmarkResultDialog);
const openInspect = () => { const tree = render(); nodes(tree).find((node) => node.props['aria-label'] === `Inspect ${workload.name} results`).props.onClick(); return render(); };
beforeEach(() => { hooks.values = []; hooks.cursor = 0; });

test('result action sits in inspector actions and details close preserves inspection and selected attempt', () => {
  let tree = openInspect();
  const actions = nodes(inspector(tree)).find((node) => node.type === DialogActions);
  const action = nodes(actions).find((node) => node.type === Button && node.props.children === 'Open result details');
  expect(action).toBeDefined();

  const choice = selector(tree).props.options[1];
  selector(tree).props.onSelectionChange(choice.id);
  tree = render();
  nodes(inspector(tree)).find((node) => node.type === Button && node.props.children === 'Open result details').props.onClick();
  tree = render();
  expect(details(tree).props.record).toEqual(choice.record);
  expect(inspector(tree).props.open).toBe(true);
  expect(inspector(tree).props.disableEnforceFocus).toBe(true);
  inspector(tree).props.onClose({}, 'escapeKeyDown');
  expect(inspector(render()).props.open).toBe(true);
  details(tree).props.onClose({}, 'escapeKeyDown');
  tree = render();
  expect(details(tree).props.record).toBeNull();
  expect(inspector(tree).props.open).toBe(true);
  expect(inspector(tree).props.disableEnforceFocus).toBe(false);
  expect(selector(tree).props.selectedId).toBe(choice.id);
  const another = selector(tree).props.options[0];
  selector(tree).props.onSelectionChange(another.id);
  tree = render();
  nodes(inspector(tree)).find((node) => node.type === Button && node.props.children === 'Open result details').props.onClick();
  expect(details(render()).props.record).toEqual(another.record);
  details(render()).props.onClose();
  inspector(render()).props.onClose({}, 'backdropClick');
  expect(inspector(render()).props.open).toBe(false);
  tree = openInspect();
  expect(selector(tree).props.selectedId).toBe('');
});

test('opening a record must not clear the inspector workload', () => {
  const tree = openInspect();
  nodes(inspector(tree)).find((node) => node.type === Button && node.props.children === 'Open result details').props.onClick();
  expect(inspector(render()).props.open).toBe(true);
});

test('selector has bounded search and controlled selection without an inline action', () => {
  const options = Array.from({ length: 60 }, (_, index) => ({ id: `attempt-${index}`, label: `Attempt ${index}` }));
  hooks.cursor = 0;
  const onSelectionChange = vi.fn();
  const props = { label: 'Published attempt', options, selectedId: '', onSelectionChange };
  const tree = ChartPointSelector(props);
  expect(nodes(tree).some((node) => node.type === Button)).toBe(false);
  const autocomplete = nodes(tree).find((node) => node.props.options === options);
  expect(autocomplete.props.filterOptions(options, { inputValue: '', getOptionLabel: (option) => option.label })).toHaveLength(50);
  expect(autocomplete.props.value).toBe(options[0]);
  autocomplete.props.onChange(null, options[1]);
  hooks.cursor = 0;
  expect(onSelectionChange).toHaveBeenCalledWith(options[1].id);
  expect(nodes(ChartPointSelector(props)).find((node) => node.props.options === options).props.value).toBe(options[0]);
  const next = ChartPointSelector({ ...props, selectedId: options[1].id });
  expect(nodes(next).find((node) => node.props.options === options).props.value).toBe(options[1]);
});

test('direct chart details do not create an inspector context', () => {
  const tree = render();
  nodes(tree).find((node) => node.type === BenchmarkHistoryChart).props.onSelectRecord({ run: runs[0], test: runs[0].tests[0] });
  expect(inspector(render()).props.open).toBe(false);
  details(render()).props.onClose();
  expect(inspector(render()).props.open).toBe(false);
});
