function clampIndex(value, itemCount) {
  return Math.max(0, Math.min(Math.round(value), Math.max(0, itemCount - 1)));
}

export function initialZoomWindow(itemCount, visibleCount, anchorIndex = -1) {
  if (itemCount <= visibleCount) return { startValue: 0, endValue: Math.max(0, itemCount - 1) };
  if (anchorIndex < 0) return { startValue: itemCount - visibleCount, endValue: itemCount - 1 };
  const desiredStart = anchorIndex - Math.floor(visibleCount / 2);
  const startValue = Math.max(0, Math.min(desiredStart, itemCount - visibleCount));
  return { startValue, endValue: startValue + visibleCount - 1 };
}

export function zoomIndexRange(zoom, itemCount) {
  if (Number.isFinite(Number(zoom.startValue)) && Number.isFinite(Number(zoom.endValue))) {
    return {
      startIndex: clampIndex(Number(zoom.startValue), itemCount),
      endIndex: clampIndex(Number(zoom.endValue), itemCount),
    };
  }
  const lastIndex = Math.max(0, itemCount - 1);
  return {
    startIndex: clampIndex((Number(zoom.start ?? 0) / 100) * lastIndex, itemCount),
    endIndex: clampIndex((Number(zoom.end ?? 100) / 100) * lastIndex, itemCount),
  };
}

export function zoomIncludingIndexes(zoom, itemCount, indexes, padding = 1) {
  const validIndexes = indexes
    .map(Number)
    .filter((index) => Number.isFinite(index) && index >= 0 && index < itemCount);
  if (validIndexes.length === 0 || itemCount <= 0) return zoom;

  const firstSelected = Math.min(...validIndexes);
  const lastSelected = Math.max(...validIndexes);
  const requiredStart = Math.max(0, firstSelected - padding);
  const requiredEnd = Math.min(itemCount - 1, lastSelected + padding);
  const { startIndex, endIndex } = zoomIndexRange(zoom, itemCount);

  if (startIndex <= requiredStart && endIndex >= requiredEnd) return zoom;
  return { startValue: requiredStart, endValue: requiredEnd };
}

export function zoomFromEvent(event, currentZoom) {
  const update = event?.batch?.at(-1) ?? event;
  if (Number.isFinite(update?.start) && Number.isFinite(update?.end)) {
    if (currentZoom.start === update.start && currentZoom.end === update.end) return currentZoom;
    return { start: update.start, end: update.end };
  }
  if (update?.startValue != null && update?.endValue != null) {
    if (currentZoom.startValue === update.startValue && currentZoom.endValue === update.endValue) return currentZoom;
    return { startValue: update.startValue, endValue: update.endValue };
  }
  return currentZoom;
}
