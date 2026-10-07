import { formatFullDate, shortSha } from '../../utils/formatters';

export function runOptionLabel(run) {
  return `${shortSha(run)} · ${run.branch} · ${formatFullDate(run.timestamp)} · ${run.runId}`;
}

export function filterRunOptions(options, { inputValue }) {
  const query = inputValue.trim().toLocaleLowerCase();
  return query ? options.filter((run) => [runOptionLabel(run), run.provenance?.rocjitsuCommitSha, run.provenance?.commitMessage, run.timestamp, run.pullRequest?.number].filter((value) => value != null).join(' ').toLocaleLowerCase().includes(query)) : options;
}

export function configurationState(candidate, baseline, comparison, target, mode) {
  const matched = comparison.comparable.length;
  const excluded = comparison.notComparable.length;
  const published = (run) => run?.configurations?.some((configuration) => configuration.target === target && configuration.mode === mode);
  const reason = !candidate ? 'Candidate attempt unavailable'
    : !baseline ? 'No reference selected'
      : !published(candidate) ? 'Candidate configuration not published'
        : !published(baseline) ? 'Reference configuration not published'
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
  return { seconds, percent: seconds !== null && baseline !== 0 ? seconds / baseline * 100 : null };
}

export function groupComparisonRows(rows, sort = 'seconds', direction = 'desc') {
  const value = (row) => rowChanges(row)[sort === 'percent' ? 'percent' : 'seconds'];
  const ordered = [...rows].sort((left, right) => {
    const a = value(left); const b = value(right);
    if (a === null && b !== null) return 1;
    if (b === null && a !== null) return -1;
    return (a !== null && b !== null ? (a - b) * (direction === 'asc' ? 1 : -1) : 0)
      || left.candidateTest.testId.localeCompare(right.candidateTest.testId);
  });
  // Contiguous suite sections retain globally sorted order. A suite may recur;
  // collecting all its rows together would move missing values ahead of measurements.
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
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? value : null;
  } catch {
    return null;
  }
}
