// ECharts rounds translate3d offsets in logical pixels. With CSS zoom, those
// offsets can land between physical pixels and blur the composited text layer.
export function positionChartTooltip(point, _params, element, _rect, { viewSize, contentSize }) {
  const parent = element.offsetParent;
  const origin = parent.getBoundingClientRect();
  const zoom = origin.width / parseFloat(getComputedStyle(parent).width);
  const ratio = window.devicePixelRatio;
  const position = point.map((value, axis) => {
    const max = Math.max(0, viewSize[axis] - contentSize[axis]);
    const preferred = value + 20 > max ? value - contentSize[axis] - 20 : value + 20;
    const start = axis === 0 ? origin.left : origin.top;
    const minPixel = Math.ceil(start * ratio);
    const maxPixel = Math.floor((start + max * zoom) * ratio);
    const pixel = Math.round((start + Math.max(0, Math.min(max, preferred)) * zoom) * ratio);
    return (Math.max(minPixel, Math.min(maxPixel, pixel)) / ratio - start) / zoom;
  });
  element.style.setProperty('--chart-tooltip-x', `${position[0]}px`);
  element.style.setProperty('--chart-tooltip-y', `${position[1]}px`);
  return position;
}
