// Shared thread-policy line encoding: ThreadPolicySwatch, chart tooltips, chart rich text and series lines.
export const POLICY_LINE_WIDTH = 3;
export const SINGLE_POLICY_DASH = [0.1, 6];
export const SWATCH_WIDTH = 22;
export const SWATCH_HEIGHT = 8;

// Colors passed here come from the dashboard palette, never publication text.
function policySvg(color, policy, attributes) {
  const dash = policy === 'single' ? ` stroke-dasharray="${SINGLE_POLICY_DASH.join(' ')}"` : '';
  const y = SWATCH_HEIGHT / 2;
  return `<svg ${attributes} viewBox="0 0 ${SWATCH_WIDTH} ${SWATCH_HEIGHT}"><line x1="1" y1="${y}" x2="${SWATCH_WIDTH - 1}" y2="${y}" stroke="${color}" stroke-width="${POLICY_LINE_WIDTH}" stroke-linecap="round"${dash}/></svg>`;
}

export function chartSwatch(color, policy) {
  return policySvg(color, policy, `class="chart-tooltip-swatch" aria-hidden="true" width="${SWATCH_WIDTH}" height="${SWATCH_HEIGHT}"`);
}

// For ECharts rich text, which cannot embed React or inline markup.
export function policySwatchImage(color, policy) {
  const svg = policySvg(color, policy, `xmlns="http://www.w3.org/2000/svg" width="${SWATCH_WIDTH}" height="${SWATCH_HEIGHT}"`);
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export function chartLineStyle(color, policy = 'default') {
  return {
    color,
    width: POLICY_LINE_WIDTH,
    type: policy === 'single' ? SINGLE_POLICY_DASH : 'solid',
    cap: 'round',
    join: 'round',
  };
}

// Dots are drawn on every measured point unless they would sit closer than
// MIN_DOT_SPACING px; then an even stride thins them so spacing stays readable at
// any chart width or range. Isolated measurements, both ends of each measured
// segment and the last slot always stay visible.
const MIN_DOT_SPACING = 140;
// Horizontal grid insets shared by both trend charts; the plot width is the chart width minus both.
export const CHART_GRID_INSETS = Object.freeze({ left: 56, right: 14 });
const CHART_PLOT_INSET = CHART_GRID_INSETS.left + CHART_GRID_INSETS.right;

export function chartPointSize(points, index, policy = 'default', valueKey = 'value', chartWidth = 0, minSpacing = MIN_DOT_SPACING) {
  const measured = (i) => Number.isFinite(points[i]?.[valueKey]);
  if (!measured(index)) return 0;
  const plotWidth = chartWidth - CHART_PLOT_INSET;
  const stride = plotWidth > 0 ? Math.max(1, Math.ceil((points.length * minSpacing) / plotWidth)) : 1;
  const visible = index % stride === 0 || index === points.length - 1
    || !measured(index - 1) || !measured(index + 1);
  return visible ? policy === 'single' ? 10 : 7 : 0;
}

export function chartPointStyle(color, surfaceColor, policy = 'default') {
  return {
    color: policy === 'single' ? surfaceColor : color,
    borderColor: policy === 'single' ? color : surfaceColor,
    borderWidth: 2,
  };
}
