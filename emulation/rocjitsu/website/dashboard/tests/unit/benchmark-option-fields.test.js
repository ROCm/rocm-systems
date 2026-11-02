import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import { BenchmarkGridPicker } from '../../src/components/benchmarks/BenchmarkPicker.jsx';
import { filterBenchmarkOptions } from '../../src/components/benchmarks/benchmarkExplorer.js';

const option = { id: 'opaque-id', name: 'Opaque workload', suite: 'Suite', operation: 'matmul', problem: { dtype: 'bf16' } };

test.each([false, 0, true, 64, 'bf16'])('benchmark search retains displayed scalar dtype %j', (dtype) => {
  const value = { ...option, problem: { dtype } };
  expect(filterBenchmarkOptions([value], String(dtype))).toEqual([value]);
});

test('benchmark picker searches and displays normalized operation and dtype fields', () => {
  expect(filterBenchmarkOptions([option], 'matmul')).toEqual([option]);
  expect(filterBenchmarkOptions([option], 'bf16')).toEqual([option]);
  const picker = BenchmarkGridPicker({ availableOptions: [option], selected: [], onChange() {} });
  const row = picker.props.renderOption({ key: option.id, role: 'option' }, option, { selected: false });
  const html = renderToStaticMarkup(row);
  expect(html).toContain('matmul · BF16');
});
