import { expect, test } from 'vitest';
import { configurationColor } from '../../src/utils/chartColors.js';
import { categoryTextColor, colorTokens, targetColors } from '../../src/theme/tokens.js';
import { THREAD_POLICIES } from '../../src/data/runSchema.js';

function luminance(hex) {
  const channels = [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16) / 255);
  return channels.map((channel) => channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4)
    .reduce((total, channel, index) => total + channel * [0.2126, 0.7152, 0.0722][index], 0);
}
const contrast = (a, b) => (Math.max(luminance(a), luminance(b)) + 0.05) / (Math.min(luminance(a), luminance(b)) + 0.05);

test.each(['constructor', 'toString', '__proto__', 'hasOwnProperty'])('literal target %s always yields a usable configuration color', (target) => {
  for (const theme of ['light', 'dark']) {
    for (const mode of THREAD_POLICIES) {
      expect(configurationColor({ target, mode }, 0, theme)).toMatch(/^#[0-9a-f]{6}$/i);
    }
  }
});

for (const theme of ['light', 'dark']) {
  const surface = colorTokens[theme].surface;
  const targets = [...Object.keys(targetColors[theme]), 'other-target'];

  // WCAG 1.4.11: chart lines and swatches need 3:1; text that shows the color needs 4.5:1.
  test(`${theme} chart and swatch colors are visible against the surface (3:1)`, () => {
    for (const target of targets) {
      const color = configurationColor({ target, mode: 'single' }, 0, theme);
      expect(contrast(color, surface), `${target}: ${color}`).toBeGreaterThanOrEqual(3);
    }
  });

  test(`${theme} category identity text is readable against the surface (4.5:1)`, () => {
    for (const target of targets) {
      const color = categoryTextColor(theme, target, undefined, 'target');
      expect(contrast(color, surface), `${target}: ${color}`).toBeGreaterThanOrEqual(4.5);
    }
  });

  test(`${theme} each target keeps its own chart color regardless of thread policy`, () => {
    const known = Object.keys(targetColors[theme]);
    const colors = known.flatMap((target) => THREAD_POLICIES.map((mode) => configurationColor({ target, mode }, 0, theme)));
    expect(new Set(colors).size).toBe(known.length);
  });
}
