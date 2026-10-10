// Pure presentation navigation; gaps are not eligible inspection points.
export function trendGapEndpoints(values) {
  const bridges = [];
  let previous = null;
  values.forEach((value, index) => {
    if (!Number.isFinite(value)) return;
    if (previous != null && index > previous + 1) bridges.push([previous, index]);
    previous = index;
  });
  return bridges;
}

export function trendKeyIndex(indexes, current, key) {
  if (key === 'Escape' || !indexes.length) return null;
  if (key === 'Home') return indexes[0];
  if (key === 'End') return indexes.at(-1);
  const position = indexes.indexOf(current);
  if (key === 'ArrowRight') return indexes[Math.min(indexes.length - 1, position + 1)];
  if (key === 'ArrowLeft') return indexes[Math.max(0, position < 0 ? indexes.length - 1 : position - 1)];
  return current;
}

export function nearestTrendIndex(history, x) {
  if (!Number.isFinite(x)) return null;
  const nearest = history.slots.reduce((nearest, slot, index) => {
    return nearest == null || Math.abs(slot.x - x) < Math.abs(history.slots[nearest].x - x) ? index : nearest;
  }, null);
  return nearest != null && history.slots[nearest].run && Number.isFinite(history.series[0]?.data[nearest]) ? nearest : null;
}

export function shouldHideTrendPointer(source) {
  return source === 'mouse' || source === 'pen';
}

export function recentRunsPage(rowCount, requestedPage) {
  const page = Math.max(0, Math.min(requestedPage, Math.ceil(rowCount / 20) - 1));
  const start = page * 20;
  return { page, start, end: Math.min(rowCount, start + 20) };
}
