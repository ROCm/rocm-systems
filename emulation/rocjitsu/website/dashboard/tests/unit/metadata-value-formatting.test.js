import { expect, test } from 'vitest';
import * as formatters from '../../src/utils/formatters.js';

test.each([
  [null, 'Not provided'],
  [undefined, 'Not provided'],
  [false, 'false'],
  [true, 'true'],
  [0, '0'],
  ['', ''],
  ['literal <value>', 'literal <value>'],
  [{ enabled: false, count: 0 }, '{"enabled":false,"count":0}'],
  [['ST', 'MT'], '["ST","MT"]'],
])('formats metadata %j without replacing literal values', (value, expected) => {
  expect(formatters.formatMetadataValue).toBeTypeOf('function');
  expect(formatters.formatMetadataValue(value)).toBe(expected);
});
