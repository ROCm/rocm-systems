import { Children, createElement, isValidElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import BenchmarkPicker, { BenchmarkGridPicker } from '../../src/components/benchmarks/BenchmarkPicker.jsx';

function descendants(node, predicate) {
  const found = [];
  function visit(element) {
    if (!isValidElement(element)) return;
    if (predicate(element)) found.push(element);
    Children.forEach(element.props.children, visit);
  }
  visit(node);
  return found;
}

function renderOption(Picker, dataType) {
  const option = { id: 'fictional-scalar', name: 'Fictional scalar workload', suite: 'Triton', problem: { operation: 'gemm', dataType } };
  const picker = Picker({ allOptions: [option], availableOptions: [option], selected: Picker === BenchmarkGridPicker ? [] : null, showingAll: false, onChange() {} });
  return picker.props.renderOption({ key: option.id, role: 'option', 'aria-selected': true, tabIndex: -1 }, option, { selected: true });
}

it('grid groups interleaved suites contiguously without mutating exact options or losing bounded search', () => {
  const options = Array.from({ length: 60 }, (_, index) => ({ id: `exact-${index}`, name: `Workload ${index}`, suite: index % 2 ? 'Llama' : 'Triton', problem: { dataType: index } }));
  const picker = BenchmarkGridPicker({ allOptions: options, availableOptions: options, selected: [], showingAll: false, onChange() {} });
  const suites = picker.props.options.map(picker.props.groupBy);
  expect(suites.filter((suite, index) => index === 0 || suite !== suites[index - 1])).toHaveLength(2);
  expect(new Set(picker.props.options)).toEqual(new Set(options));
  expect(options[0].id).toBe('exact-0');
  expect(picker.props.filterOptions(picker.props.options, { inputValue: '' })).toHaveLength(50);
  expect(picker.props.filterOptions(picker.props.options, { inputValue: 'Workload 59' })).toEqual([options[59]]);
});

it('grid popup and long chip sets have viewport-bounded local scrolling', () => {
  const picker = BenchmarkGridPicker({ allOptions: [], availableOptions: [], selected: [], showingAll: false, onChange() {} });
  expect(picker.props.limitTags).toBe(-1);
  expect(picker.props.sx['& .MuiAutocomplete-inputRoot']?.overflowY).toBe('auto');
  expect(picker.props.sx['& .MuiAutocomplete-inputRoot'].maxHeight).toContain('dvh');
  const listbox = picker.props.slotProps.listbox.sx;
  expect(listbox.maxHeight).toContain('--benchmark-listbox-height');
  expect(listbox.overflowY).toBe('auto');
  const modifier = picker.props.slotProps.popper.modifiers.find((item) => item.name === 'benchmarkViewport');
  const values = {};
  const state = { elements: { reference: { ownerDocument: { defaultView: { innerHeight: 600 } }, getBoundingClientRect: () => ({ top: 200, bottom: 372 }) }, popper: { style: { setProperty: (key, value) => { values[key] = value; } }, offsetHeight: 212 } }, rects: { popper: { height: 340 } } };
  modifier.fn({ state });
  expect(values['--benchmark-listbox-height']).toBe('220px');
  expect(state.rects.popper.height).toBe(212);
  state.elements.reference.getBoundingClientRect = () => ({ top: 350, bottom: 520 });
  modifier.fn({ state });
  expect(values['--benchmark-listbox-height']).toBe('340px');
});

it('UI-005 keeps workload options named with no independently focusable nested checkbox', () => {
  const row = renderOption(BenchmarkGridPicker, 'fp16');
  const html = renderToStaticMarkup(row);
  expect(html).toContain('role="option"');
  expect(html).toContain('aria-selected="true"');
  expect(html).toContain('Fictional scalar workload');
  expect(html).not.toMatch(/<(?:input|button)\b/);
  expect(html).toMatch(/<span\b[^>]*aria-hidden="true"/);
});

it.each([16, true, false, 0, null, 'fp16'])('UI-001 renders dataType %j safely, formatting only strings', (dataType) => {
  for (const Picker of [BenchmarkGridPicker, BenchmarkPicker]) {
    const option = renderOption(Picker, dataType);
    const description = descendants(option, (element) => 'secondary' in element.props)[0].props.secondary;
    expect(description).toBe(dataType === null ? 'gemm' : `gemm · ${typeof dataType === 'string' ? dataType.toUpperCase() : String(dataType)}`);
    expect(renderToStaticMarkup(createElement(() => option))).toContain('Fictional scalar workload');
  }
});
