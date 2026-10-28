import { expect, test } from 'vitest';
import { configurationColor } from '../../src/utils/chartColors.js';
import { colorTokens, targetColors } from '../../src/theme/tokens.js';
import { THREAD_POLICIES } from '../../src/data/runSchema.js';

function luminance(hex) {
  const channels = [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16) / 255);
  return channels.map((channel) => channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4)
    .reduce((total, channel, index) => total + channel * [0.2126, 0.7152, 0.0722][index], 0);
}

test.each(['constructor', 'toString', '__proto__', 'hasOwnProperty'])('literal target %s always yields a usable configuration color', (target) => {
  for (const theme of ['light', 'dark']) {
    for (const mode of THREAD_POLICIES) {
      expect(configurationColor({ target, mode }, 0, theme)).toMatch(/^#[0-9a-f]{6}$/i);
    }
  }
});

for (const theme of ['light', 'dark']) {
  test(`${theme} Overview configuration text retains normal-text contrast including fallback targets`, () => {
    const background = luminance(colorTokens[theme].surface);
    const targets = Object.keys(targetColors[theme]);
    for (const target of [...targets, 'other-target']) {
      for (const mode of THREAD_POLICIES) {
        for (let index = 0; index < 5; index++) {
          const color = configurationColor({ target, mode }, index, theme);
          const foreground = luminance(color);
          const contrast = (Math.max(background, foreground) + 0.05) / (Math.min(background, foreground) + 0.05);
          expect(contrast, `${target}/${mode} index ${index}: ${color}`).toBeGreaterThanOrEqual(4.5);
        }
      }
    }
    const identities = targets.flatMap((target) => THREAD_POLICIES.map((mode) => configurationColor({ target, mode }, 0, theme)));
    expect(new Set(identities).size).toBe(targets.length * THREAD_POLICIES.length);
  });
}
