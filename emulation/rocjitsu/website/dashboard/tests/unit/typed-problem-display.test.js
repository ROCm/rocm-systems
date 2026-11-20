import { expect, test } from 'vitest';
import { problemDetails } from '../../src/data/problemDetails.js';

test('typed workload values are display text without mutating arrays, objects or explicit null', () => {
  const problem = { shape: [1, 2], options: { nested: [false, null] }, note: null, literal: '[1,2]' };
  const before = structuredClone(problem);
  expect(problemDetails(problem).map(({ key, value }) => [key, value])).toEqual([
    ['shape', '[1,2]'], ['options', '{"nested":[false,null]}'], ['note', 'null'], ['literal', '[1,2]'],
  ]);
  expect(problem).toEqual(before);
});
