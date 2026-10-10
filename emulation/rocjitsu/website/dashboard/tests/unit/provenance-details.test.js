import { expect, test } from 'vitest';
import { provenanceDetails } from '../../src/data/provenance.js';

test('generic provenance preserves distinct keys even when labels repeat, including empty scalar strings', () => {
  expect(provenanceDetails({ details: [
    { key: 'sdk-a', label: 'SDK', value: 'one' }, { key: 'sdk-b', label: 'SDK', value: 'two' }, { key: 'empty', label: 'Empty fact', value: '' },
  ] })).toEqual([
    { key: 'sdk-a', label: 'SDK', value: 'one' }, { key: 'sdk-b', label: 'SDK', value: 'two' }, { key: 'empty', label: 'Empty fact', value: '' },
  ]);
});
