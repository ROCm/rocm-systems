import { Children, createElement, isValidElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import { BenchmarkGridPicker } from '../../src/components/benchmarks/BenchmarkPicker.jsx';

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

function renderOption(dataType) {
  const option = { id: 'fictional-scalar', name: 'Fictional scalar workload', suite: 'Triton', operation: 'gemm', problem: { dtype: dataType } };
  const picker = BenchmarkGridPicker({ availableOptions: [option], selected: [], onChange() {} });
  return picker.props.renderOption({ key: option.id, role: 'option', 'aria-selected': true, tabIndex: -1 }, option, { selected: true });
}

it('picker uses the global cap even when a caller requests more, and retains smaller bounds', () => {
  const options = Array.from({ length: 10 }, (_, index) => ({ id: `${index}`, name: `Workload ${index}`, suite: 'Triton' }));
  for (const [requested, effective] of [[undefined, 8], [9, 8], [3, 3]]) {
    let chosen;
    const picker = BenchmarkGridPicker({ availableOptions: options, selected: options.slice(0, effective), maxSelected: requested, onChange: (next) => { chosen = next; } });
    expect(picker.props.getOptionDisabled(options[effective])).toBe(true);
    expect(picker.props.getOptionDisabled(options[0])).toBe(false);
    picker.props.onChange(null, [...options, options[0]]);
    expect(chosen).toEqual(options.slice(0, effective));
    expect(picker.props.renderInput({}).props.helperText).toContain(`${effective} of ${effective} benchmarks selected`);
  }
});

it('grid groups interleaved suites contiguously without mutating exact options or losing bounded search', () => {
  const options = Array.from({ length: 60 }, (_, index) => ({ id: `exact-${index}`, name: `Workload ${index}`, suite: index % 2 ? 'Llama' : 'Triton', problem: { dataType: index } }));
  const picker = BenchmarkGridPicker({ availableOptions: options, selected: [], onChange() {} });
  const suites = picker.props.options.map(picker.props.groupBy);
  expect(suites.filter((suite, index) => index === 0 || suite !== suites[index - 1])).toHaveLength(2);
  expect(new Set(picker.props.options)).toEqual(new Set(options));
  expect(options[0].id).toBe('exact-0');
  expect(picker.props.filterOptions(picker.props.options, { inputValue: '' })).toHaveLength(50);
  expect(picker.props.filterOptions(picker.props.options, { inputValue: 'Workload 59' })).toEqual([options[59]]);
});

it('grid popup and long chip sets have viewport-bounded local scrolling', () => {
  const picker = BenchmarkGridPicker({ availableOptions: [], selected: [], onChange() {} });
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
  const row = renderOption('fp16');
  const html = renderToStaticMarkup(row);
  expect(html).toContain('role="option"');
  expect(html).toContain('aria-selected="true"');
  expect(html).toContain('Fictional scalar workload');
  expect(html).not.toMatch(/<(?:input|button)\b/);
  expect(html).toMatch(/<span\b[^>]*aria-hidden="true"/);
});

it.each([16, true, false, 0, null, 'fp16'])('UI-001 renders dataType %j safely, formatting only strings', (dataType) => {
  const option = renderOption(dataType);
  const description = descendants(option, (element) => 'secondary' in element.props)[0].props.secondary;
  const markup = renderToStaticMarkup(description);
  expect(markup).toContain('fictional-scalar');
  expect(markup).toContain(dataType === null ? 'gemm' : `gemm · ${typeof dataType === 'string' ? dataType.toUpperCase() : String(dataType)}`);
  expect(renderToStaticMarkup(createElement(() => option))).toContain('Fictional scalar workload');
});
