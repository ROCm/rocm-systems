export const TARGET_COLORS = {
  gfx1250: '#2166C1',
  gfx950: '#C25430',
  gfx1201: '#7450B8',
};

const targetFallbackColors = ['#2166C1', '#C25430', '#7450B8', '#14826F', '#B7791F'];

export function targetColor(target, index = 0) {
  return TARGET_COLORS[target] ?? targetFallbackColors[index % targetFallbackColors.length];
}
