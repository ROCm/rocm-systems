import { expect, test } from 'vitest';
import { hslToHex, softenColor } from '../../src/utils/colorMath.js';

test.each([
  [[0, 1, 0.5], '#ff0000'], [[60, 1, 0.5], '#ffff00'], [[120, 1, 0.5], '#00ff00'],
  [[180, 1, 0.5], '#00ffff'], [[240, 1, 0.5], '#0000ff'], [[300, 1, 0.5], '#ff00ff'],
  [[0, 0, 0.5], '#808080'], [[0, 1, 0], '#000000'], [[0, 1, 1], '#ffffff'],
])('hslToHex(%j) is %s', ([hue, saturation, lightness], expected) => {
  expect(hslToHex(hue, saturation, lightness)).toBe(expected);
});

test('softenColor keeps the hue and lowers saturation, with literal results', () => {
  expect(softenColor('#ff0000', 0.5, 0)).toBe('#bf4040');
  expect(softenColor('#0000ff', 0, 0)).toBe('#808080');
  expect(softenColor('#808080')).toBe('#7b7b7b');
});
