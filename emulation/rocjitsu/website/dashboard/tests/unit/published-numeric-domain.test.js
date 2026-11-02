import { describe, expect, it } from 'vitest';
import { parsePublishedJson } from '../../src/data/publishedJson.js';
import { isJsonValue } from '../../src/data/jsonValues.js';

describe('supported publication numeric domain', () => {
  it.each(['9007199254740993', '9007199254740992', '9007199254740993.0', '9.007199254740993e15', '1e308', '1e-400', '9007199254740991.1', '-9.007199254740993e15'])('rejects unsupported text token %s before export', (token) => {
    expect(() => parsePublishedJson(`{"problem":{"nested":[${token}]}}`)).toThrow(/numeric/i);
  });
  it.each(['9007199254740991.0', '9.007199254740991e15', '0e999999', '-0e-999999'])('accepts safe numeric spelling %s', (token) => {
    expect(parsePublishedJson(`[${token}]`)).toEqual(JSON.parse(`[${token}]`));
  });
  it('rejects unsafe already-parsed recursive values', () => {
    expect(isJsonValue({ nested: [9007199254740992] })).toBe(false);
  });
  it('preserves accepted numbers and other JSON types', () => {
    const text = '{"a":[9007199254740991,-9007199254740991,1.25,5e-324,0,null,true,"9007199254740993"]}';
    expect(parsePublishedJson(text)).toEqual(JSON.parse(text));
    expect(isJsonValue(JSON.parse(text))).toBe(true);
  });
});
