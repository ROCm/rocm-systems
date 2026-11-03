// Colors passed here come from the dashboard palette, never publication text.
export function chartSwatch(color, policy) {
  return `<svg class="chart-tooltip-swatch" viewBox="0 0 22 8" aria-hidden="true"><line x1="1" y1="4" x2="21" y2="4" stroke="${color}" stroke-width="3" stroke-linecap="round"${policy === 'single' ? ' stroke-dasharray="0.1 6"' : ''}/></svg>`;
}

export function chartLineStyle(color, policy = 'default') {
  return {
    color,
    width: 3,
    type: policy === 'single' ? [0.1, 6] : 'solid',
    cap: 'round',
    join: 'round',
  };
}

// Dots are drawn on every measured point unless they would sit closer than
// MIN_DOT_SPACING px; then an even stride thins them so spacing stays readable at
// any chart width or range. Isolated measurements, both ends of each measured
// segment and the last slot always stay visible.
export const MIN_DOT_SPACING = 140;
// Both charts use grid left 56 + right 14.
export const CHART_PLOT_INSET = 70;

export function chartPointSize(points, index, policy, valueKey = 'value', chartWidth = 0, minSpacing = MIN_DOT_SPACING) {
  const measured = (i) => Number.isFinite(points[i]?.[valueKey]);
  if (!measured(index)) return 0;
  const plotWidth = chartWidth - CHART_PLOT_INSET;
  const stride = plotWidth > 0 ? Math.max(1, Math.ceil((points.length * minSpacing) / plotWidth)) : 1;
  const visible = index % stride === 0 || index === points.length - 1
    || !measured(index - 1) || !measured(index + 1);
  return visible ? policy === 'single' ? 10 : 7 : 0;
}

export function chartPointStyle(color, surfaceColor, policy = 'single') {
  return {
    color: policy === 'single' ? surfaceColor : color,
    borderColor: policy === 'single' ? color : surfaceColor,
    borderWidth: 2,
  };
}
