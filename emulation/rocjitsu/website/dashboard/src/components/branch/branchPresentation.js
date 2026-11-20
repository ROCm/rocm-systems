import { isSafeHttpUrl } from '../../utils/externalUrls';
import { formatFullDate, shortSha } from '../../utils/formatters';
import { hasConfiguration } from '../../data/runSchema';
import { runSearchFields } from '../shared/runOptionPresentation';

export function runOptionLabel(run) {
  return `${shortSha(run)} · ${run.branch} · ${formatFullDate(run.timestamp)} · ${run.runId}`;
}

export function filterRunOptions(options, { inputValue }) {
  const query = inputValue.trim().toLowerCase();
  return query ? options.filter((run) => [runOptionLabel(run), ...runSearchFields(run), run.pullRequest?.number]
    .filter((value) => value != null).join(' ').toLowerCase().includes(query)) : options;
}

export function configurationState(candidate, baseline, comparison, target, mode) {
  const matched = comparison.comparable.length;
  const excluded = comparison.notComparable.length;
  const reason = !candidate ? 'Candidate attempt unavailable'
    : !baseline ? 'No reference selected'
      : !hasConfiguration(candidate, target, mode) ? 'Candidate configuration not published'
        : !hasConfiguration(baseline, target, mode) ? 'Reference configuration not published'
          : !matched ? (excluded ? 'No matched completed benchmarks' : 'No benchmarks in the selected suites or search') : null;
  return { matched, excluded, available: reason === null, reason,
    deltaSeconds: reason ? null : comparison.candidateDuration - comparison.baselineDuration,
    deltaPercent: reason ? null : comparison.aggregateDelta,
  };
}

export function rowChanges(row) {
  const baseline = row.baselineTest?.durationSeconds;
  const candidate = row.candidateTest?.durationSeconds;
  const seconds = Number.isFinite(baseline) && Number.isFinite(candidate) ? candidate - baseline : null;
  const percent = seconds !== null && baseline !== 0 ? seconds / baseline * 100 : null;
  return { seconds, percent: Number.isFinite(percent) ? percent : null };
}

export function groupComparisonRows(rows, sort = 'seconds', direction = 'desc') {
  const value = (row) => rowChanges(row)[sort === 'percent' ? 'percent' : 'seconds'];
  const ordered = [...rows].sort((left, right) => {
    const suiteOrder = left.candidateTest.suite.localeCompare(right.candidateTest.suite);
    if (suiteOrder) return suiteOrder;
    const a = value(left); const b = value(right);
    if (a === null && b !== null) return 1;
    if (b === null && a !== null) return -1;
    return (a !== null && b !== null ? (a - b) * (direction === 'asc' ? 1 : -1) : 0)
      || left.candidateTest.testId.localeCompare(right.candidateTest.testId);
  });
  // One alphabetical section per suite; sort measured changes within each suite,
  // retaining unavailable values last regardless of the requested direction.
  const groups = [];
  for (const row of ordered) {
    const suite = row.candidateTest.suite;
    if (groups.at(-1)?.suite !== suite) groups.push({ suite, rows: [] });
    groups.at(-1).rows.push(row);
  }
  return groups;
}

export function safeExternalUrl(value) {
  if (typeof value !== 'string' || !/^https?:\/\//i.test(value)) return null;
  try {
    const url = new URL(value);
    return isSafeHttpUrl(url) ? value : null;
  } catch {
    return null;
  }
}
