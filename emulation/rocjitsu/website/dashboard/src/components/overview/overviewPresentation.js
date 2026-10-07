// Pure presentation navigation; gaps are not eligible inspection points.
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

export function recentVisibleRange(rowCount, scrollTop = 0) {
  if (!rowCount) return { start: 0, end: 0 };
  const first = Math.min(Math.max(0, rowCount - 5), Math.max(0, Math.floor(scrollTop / 60)));
  return { start: first + 1, end: Math.min(rowCount, first + 5) };
}
