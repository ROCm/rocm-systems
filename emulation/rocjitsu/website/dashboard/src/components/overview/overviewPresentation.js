import { formatDuration, formatFullDate } from '../../utils/formatters.js';

// Keyboard callers supply chronological slots; pointer callers stay finite-only.
export function trendEmptyMessage(projection) {
  if (!projection.rows.length) return 'Select targets and thread policies to compare performance.';
  if (!projection.attempts.length) return 'No develop runs in the selected range/benchmark set.';
  if (projection.rows.every((row) => !row.cohortIds.length)) {
    return 'No shared workloads. Select configurations and benchmark families with shared workloads.';
  }
  if (projection.range === '1W' && projection.rows.some((row) => !row.cohortIds.length)) {
    return 'No shared workloads for one or more selected configurations.';
  }
  if (projection.range === '1W' && !projection.attempts.some((attempt) => attempt.eligible)) {
    return 'No complete runs for selected comparison in this week.';
  }
  return 'Relative speedup unavailable: no positive valid reference in the selected range.';
}

export function trendSlotStatus(projection, index) {
  const slot = projection.slots[index];
  if (!slot) return '';
  const day = slot.dayKey ?? projection.dayKeys?.[Math.round(slot.x)];
  const identity = [day ? `${day} UTC` : 'Date unavailable'];
  const noRun = projection.range === '1W' ? 'No eligible plotted run' : 'No published run in this view';
  if (slot.run) {
    identity.push(`Attempt ${slot.run.runId}`);
    if (slot.run.timestamp) identity.push(`Run time: ${formatFullDate(slot.run.timestamp)}`);
  } else identity.push(noRun);
  const reasons = { failed: 'Failed workload', timeout: 'Timed out', 'configuration-absent': 'Configuration absent',
    'missing-result': 'Missing required result', 'no-comparable-workloads': 'No shared workloads',
    'invalid-duration': 'Invalid duration', 'invalid-total': 'Invalid total', 'no-run': noRun };
  return identity.concat(projection.rows.map((row) => {
    const point = row.points[index];
    const value = Number.isFinite(point?.speedup)
      ? `${point.speedup.toFixed(3)}× · ${formatDuration(point.durationSeconds)}`
      : `Unavailable — ${!slot.run ? noRun : reasons[point?.reason] ?? 'Relative speedup unavailable'}`;
    return `${row.target} ${row.mode}: ${value}`;
  })).join(' · ');
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
  return nearest != null && history.slots[nearest].run && history.rows.some((row) => Number.isFinite(row.points[nearest]?.speedup)) ? nearest : null;
}

const RECENT_RUNS_PAGE_SIZE = 20;

export function recentRunsPage(rowCount, requestedPage) {
  const pageSize = RECENT_RUNS_PAGE_SIZE;
  const pageCount = Math.ceil(rowCount / pageSize);
  const page = Math.max(0, Math.min(requestedPage, pageCount - 1));
  const start = page * pageSize;
  return { page, start, end: Math.min(rowCount, start + pageSize), pageCount, pageSize };
}
