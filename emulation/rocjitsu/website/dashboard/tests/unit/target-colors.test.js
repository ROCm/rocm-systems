import { describe, expect, test } from 'vitest';
import { generateTargetColors, targetColor, targetHues } from '../../src/theme/targetColors.js';
import { createDashboardTheme } from '../../src/theme/theme.js';
import { colorTokens } from '../../src/theme/tokens.js';

function luminance(hex) {
  const channels = [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16) / 255);
  return channels.map((channel) => channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4)
    .reduce((total, channel, index) => total + channel * [0.2126, 0.7152, 0.0722][index], 0);
}
const contrast = (a, b) => (Math.max(luminance(a), luminance(b)) + 0.05) / (Math.min(luminance(a), luminance(b)) + 0.05);
const hueOf = (hex) => {
  const [r, g, b] = [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16) / 255);
  const max = Math.max(r, g, b); const delta = max - Math.min(r, g, b);
  if (!delta) return 0;
  const sector = max === r ? ((g - b) / delta + 6) % 6 : max === g ? (b - r) / delta + 2 : (r - g) / delta + 4;
  return sector * 60;
};
const gap = (a, b) => Math.min(Math.abs(a - b), 360 - Math.abs(a - b));
const named = (count) => Array.from({ length: count }, (_, index) => `target-${String(index).padStart(2, '0')}`);

for (const mode of ['light', 'dark']) {
  const surface = colorTokens[mode].surface;

  describe(`${mode} generated target colors`, () => {
    test('the first two targets keep the original blue and orange', () => {
      const colors = generateTargetColors(mode, ['gfx950', 'gfx1250']);
      expect(colors.get('gfx1250')).toBe(colorTokens[mode].interactive);
      expect(colors.get('gfx950')).toBe(colorTokens[mode].seriesSecondary);
    });

    test.each([2, 3, 6, 12, 24])('%i targets get distinct colors', (count) => {
      const colors = [...generateTargetColors(mode, named(count)).values()];
      expect(new Set(colors).size).toBe(count);
    });

    test.each([3, 6, 12, 24])('every generated color of %i targets is readable on the surface (4.5:1)', (count) => {
      for (const [target, color] of generateTargetColors(mode, named(count))) {
        expect(color, target).toMatch(/^#[0-9a-f]{6}$/i);
        expect(contrast(color, surface), `${target}: ${color}`).toBeGreaterThanOrEqual(4.5);
      }
    });

    test('generated hues stay apart from each other and from the seeded blue and orange', () => {
      const hues = [...generateTargetColors(mode, named(8)).values()].map(hueOf);
      for (let a = 0; a < hues.length; a += 1) for (let b = a + 1; b < hues.length; b += 1) {
        expect(gap(hues[a], hues[b]), `${a} vs ${b}`).toBeGreaterThanOrEqual(15);
      }
    });

    test('a target keeps its color when the list is unchanged, whatever order it was published in', () => {
      const forward = generateTargetColors(mode, ['gfx1100', 'gfx942', 'gfx950', 'gfx1250']);
      const reversed = generateTargetColors(mode, ['gfx1250', 'gfx950', 'gfx942', 'gfx1100']);
      expect([...forward]).toEqual([...reversed]);
    });

    test('the theme exposes the generated colors', () => {
      const theme = createDashboardTheme(mode, ['gfx950', 'gfx1250', 'gfx942']);
      expect(Object.keys(theme.palette.targetColors).sort()).toEqual(['gfx1250', 'gfx942', 'gfx950']);
      expect(new Set(Object.values(theme.palette.targetColors)).size).toBe(3);
    });
  });
}

test('targetHues seeds blue and orange and never loops forever on a large list', () => {
  expect(targetHues(2)).toEqual([219, 25]);
  // The seeds must stay the hues of the two palette colors the first two targets actually get.
  const [blue, orange] = targetHues(2);
  for (const mode of ['light', 'dark']) {
    expect(gap(hueOf(colorTokens[mode].interactive), blue), `${mode} blue`).toBeLessThan(8);
    expect(gap(hueOf(colorTokens[mode].seriesSecondary), orange), `${mode} orange`).toBeLessThan(8);
  }
  expect(targetHues(60)).toHaveLength(60);
});

// Inherited object keys must never resolve to Object.prototype members: they are ordinary published targets.
test.each(['constructor', 'toString', '__proto__', 'hasOwnProperty'])('literal target %s gets a usable, distinct color', (target) => {
  for (const mode of ['light', 'dark']) {
    const colors = generateTargetColors(mode, [target, 'gfx950', 'gfx1250']);
    expect(colors.get(target)).toMatch(/^#[0-9a-f]{6}$/i);
    expect(new Set(colors.values()).size).toBe(3);
    const theme = createDashboardTheme(mode, [target, 'gfx950', 'gfx1250']);
    expect(Object.hasOwn(theme.palette.targetColors, target)).toBe(true);
  }
});

test('a target the theme was not given falls back to neutral text, never another target\'s color', () => {
  const theme = createDashboardTheme('light', ['gfx1250', 'gfx950']);
  expect(targetColor(theme, 'not-published')).toBe(theme.palette.text.secondary);
  expect(Object.values(theme.palette.targetColors)).not.toContain(theme.palette.text.secondary);
});

test.each(['constructor', 'toString', 'hasOwnProperty', '__proto__'])('inherited object member %s is not a target and falls back to neutral text', (name) => {
  const theme = createDashboardTheme('light', ['gfx1250', 'gfx950']);
  expect(targetColor(theme, name)).toBe(theme.palette.text.secondary);
});
