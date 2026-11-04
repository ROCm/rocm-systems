// ECharts rounds translate3d offsets in logical pixels. On a display with a fractional device
// pixel ratio, those offsets can land between physical pixels and blur the composited text layer,
// so the tooltip is placed on a whole physical pixel.
export function positionChartTooltip(point, _params, element, _rect, { viewSize, contentSize }) {
  const parent = element.offsetParent;
  // A queued tooltip can outlive its chart during navigation.
  if (!parent) return [0, 0];
  const origin = parent.getBoundingClientRect();
  const ratio = window.devicePixelRatio;
  const position = point.map((value, axis) => {
    const start = axis === 0 ? origin.left : origin.top;
    const viewport = axis === 0 ? window.innerWidth : window.innerHeight;
    const min = Math.max(0, 8 - start);
    const max = Math.max(min, Math.min(viewSize[axis] - contentSize[axis], viewport - 8 - start - contentSize[axis]));
    const preferred = value + 16 > max ? value - contentSize[axis] - 16 : value + 16;
    const minPixel = Math.ceil((start + min) * ratio);
    const maxPixel = Math.floor((start + max) * ratio);
    const pixel = Math.round((start + Math.max(min, Math.min(max, preferred))) * ratio);
    return Math.max(minPixel, Math.min(maxPixel, pixel)) / ratio - start;
  });
  element.style.setProperty('--chart-tooltip-x', `${position[0]}px`);
  element.style.setProperty('--chart-tooltip-y', `${position[1]}px`);
  return position;
}
