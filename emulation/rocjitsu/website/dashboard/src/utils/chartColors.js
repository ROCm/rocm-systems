import { categoryColor } from '../theme/tokens.js';

export function targetColor(target, index = 0, themeMode = 'light') {
  const mode = themeMode === 'dark' ? 'dark' : 'light';
  return categoryColor(mode, target ?? `unknown-target-${index}`, undefined, 'target');
}

// Target owns the hue; thread policy is distinguished by line style.
export function configurationColor(row, index = 0, themeMode = 'light') {
  return targetColor(row.target, index, themeMode);
}
