import { colorTokens } from './tokens.js';
import { hslToHex } from '../utils/colorMath.js';

// Target chart colors follow list position: targets are sorted, and the first FIXED_TARGET_COUNT (seven) take a fixed palette, so
// those colors never change when targets are appended. The first two are the original blue and orange; the other five
// are the colors the generator below produced for those positions. Later targets are generated, so a new target never
// reuses another target's color and no assignment has to be added by hand. A target that sorts ahead of the others
// shifts every position after it; that is accepted.
const GOLDEN_ANGLE = 137.508;
const MIN_HUE_GAP = 30;
const SEED_HUES = [219, 25];
const SATURATION = 0.72;
// Text on tags and rows uses the same color, so every generated color clears WCAG 4.5:1 on the surface.
const MIN_CONTRAST = 4.5;

const channels = (hex) => [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16));
const luminance = (hex) => channels(hex).map((value) => value / 255)
  .map((value) => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4)
  .reduce((total, value, index) => total + value * [0.2126, 0.7152, 0.0722][index], 0);
const contrast = (a, b) => (Math.max(luminance(a), luminance(b)) + 0.05) / (Math.min(luminance(a), luminance(b)) + 0.05);
const hueGap = (a, b) => Math.min(Math.abs(a - b), 360 - Math.abs(a - b));

// Darkest (light theme) or lightest (dark theme) step that is still readable: keeps the hue vivid.
function readableHue(hue, mode) {
  const surface = colorTokens[mode].surface;
  const direction = mode === 'dark' ? 1 : -1;
  let lightness = mode === 'dark' ? 0.6 : 0.52;
  for (let step = 0; step < 30; step += 1, lightness += direction * 0.02) {
    const color = hslToHex(hue, SATURATION, lightness);
    if (contrast(color, surface) >= MIN_CONTRAST) return color;
  }
  return hslToHex(hue, SATURATION, lightness);
}

// Positions 2 to 6, per theme. Literal on purpose: a change to the generator below must not recolor these.
const FIXED_LITERALS = Object.freeze({
  light: Object.freeze(['#158463', '#ca21ca', '#5d7b14', '#dd2c42', '#158428']),
  dark: Object.freeze(['#50e2b8', '#e250e2', '#b8e250', '#e4586a', '#50e268']),
});
// Positions 0 and 1 are the palette's blue and orange, so a change to those tokens reaches the charts.
const FIXED_TARGET_COLORS = Object.freeze({
  light: Object.freeze([colorTokens.light.interactive, colorTokens.light.seriesSecondary, ...FIXED_LITERALS.light]),
  dark: Object.freeze([colorTokens.dark.interactive, colorTokens.dark.seriesSecondary, ...FIXED_LITERALS.dark]),
});
export const FIXED_TARGET_COUNT = FIXED_TARGET_COLORS.light.length;

// Hue k steps by the golden angle from the second seed, skipping hues too close to one already chosen.
export function targetHues(count) {
  const hues = SEED_HUES.slice(0, count);
  for (let k = 1; hues.length < count; k += 1) {
    const hue = (SEED_HUES[1] + GOLDEN_ANGLE * k) % 360;
    if (hues.every((chosen) => hueGap(chosen, hue) >= MIN_HUE_GAP)) hues.push(hue);
    if (k > 720) hues.push(hue); // every gap exhausted: accept the nearest rather than loop forever
  }
  return hues;
}

// Map of target -> color for one theme mode. The first two are the palette tokens; positions 2 to 6 are fixed;
// later positions are generated.
export function generateTargetColors(mode, targets = []) {
  const fixed = FIXED_TARGET_COLORS[mode];
  const unique = [...new Set(targets)].sort();
  const hues = targetHues(unique.length);
  return new Map(unique.map((target, index) => [target, index < fixed.length ? fixed[index] : readableHue(hues[index], mode)]));
}

// A target the theme was not given (nothing published for it) gets a neutral color, never another target's.
export function targetColor(theme, target) {
  const colors = theme.palette.targetColors;
  return colors && Object.hasOwn(colors, target) ? colors[target] : theme.palette.text.secondary;
}

// Tiny identity text needs stronger contrast than its decorative marker/plot hue.
export function targetTextColor(theme, target) {
  const color = targetColor(theme, target);
  const palette = theme.palette.dashboard ?? colorTokens[theme.palette.mode];
  return color === palette.seriesSecondary ? palette.seriesSecondaryText : color;
}
