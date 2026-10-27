import { targetColors } from '../theme/tokens.js';

const targetFallbackColors = {
  light: [targetColors.light.gfx1250, targetColors.light.gfx950, targetColors.light.gfx1201, '#14826F', '#8a3800'],
  dark: [targetColors.dark.gfx1250, targetColors.dark.gfx950, targetColors.dark.gfx1201, '#3ddbd9', '#f1c21b'],
};

export function targetColor(target, index = 0, themeMode = 'light') {
  const mode = themeMode === 'dark' ? 'dark' : 'light';
  const colors = targetColors[mode];
  const fallback = targetFallbackColors[mode];
  return Object.hasOwn(colors, target) ? colors[target] : fallback[index % fallback.length];
}

// Deliberate policy-specific hues distinguish the default-policy configuration.
const defaultConfigurationColors = {
  light: { gfx1250: '#14826F', gfx950: '#9f1853', gfx1201: '#725600' },
  dark: { gfx1250: '#3ddbd9', gfx950: '#ff7eb6', gfx1201: '#f1c21b' },
};

export function configurationColor(row, index = 0, themeMode = 'light') {
  const colors = defaultConfigurationColors[themeMode === 'dark' ? 'dark' : 'light'];
  return row.mode === 'default' && Object.hasOwn(colors, row.target)
    ? colors[row.target] : targetColor(row.target, row.mode === 'default' ? index + 2 : index, themeMode);
}
