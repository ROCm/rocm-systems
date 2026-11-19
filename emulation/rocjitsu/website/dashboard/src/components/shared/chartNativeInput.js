import { normalizeEvent } from 'zrender/lib/core/event.js';

// ZRender 6.1's native-offset shortcut ignores CSS scaling. Keep its metadata
// normalization, then correct coordinates at its borderless painter viewport.
const events = 'click dblclick contextmenu wheel mousewheel mousedown mouseup mousemove mouseout pointerdown pointerup pointermove pointerout touchstart touchmove touchend'.split(' ');

export function bindChartNativeInput(chart) {
  const viewport = !chart.isDisposed() && chart.getZr()?.painter?.getViewportRoot();
  if (!viewport) return () => {};
  const cleanup = () => events.forEach((type) => viewport.removeEventListener(type, capture, { capture: true }));
  const capture = (event) => {
    if (chart.isDisposed()) { cleanup(); return; }
    if (event.zrX != null) return;
    const { left, top, width: renderedWidth, height: renderedHeight } = viewport.getBoundingClientRect();
    const width = chart.getWidth(); const height = chart.getHeight();
    const point = event.type.startsWith('touch')
      ? (event.type === 'touchend' ? event.changedTouches?.[0] : event.targetTouches?.[0]) : event;
    if (![left, top, width, height, renderedWidth, renderedHeight, point?.clientX, point?.clientY].every(Number.isFinite)
      || Math.min(width, height, renderedWidth, renderedHeight) <= 0
      || (width === renderedWidth && height === renderedHeight)) return;
    normalizeEvent(viewport, event);
    event.zrX = (point.clientX - left) * width / renderedWidth;
    event.zrY = (point.clientY - top) * height / renderedHeight;
  };
  events.forEach((type) => viewport.addEventListener(type, capture, { capture: true, passive: true }));
  return cleanup;
}
