import { formatDuration, formatFullDate, shortSha } from '../../utils/formatters';
import { commitTimestampFor } from '../../data/runOrdering';
import { selectBenchmarkCatalog } from '../../data/selectors';
import { historyRangeById } from '../../config/historyRanges';

export function benchmarkResultChoices(viewModel) {
  return viewModel.runs.flatMap((run, index) => viewModel.series.map((series) => {
    const record = series.records[index];
    const status = record?.test.status ?? 'unavailable';
    return {
      id: `${series.name}:${run.runId}`, run, record, target: series.target ?? series.name, mode: series.mode, status,
      label: `${series.name} · ${shortSha(run)} · ${status}${status === 'completed' ? ` · ${formatDuration(record.test.durationSeconds)}` : ''} · ${formatFullDate(run.timestamp)} · ${run.runId}`,
    };
  })).reverse();
}

// Local presentation helpers for Benchmark Explorer.
export function boundedGridSelection(options, maximum = 8) {
  return [...new Map(options.map((option) => [option.id, option])).values()].slice(0, Math.min(maximum, 8));
}

export function filterBenchmarkOptions(options, query) {
  const normalized = query.trim().toLowerCase();
  return options.filter((option) => [option.name, option.suite, option.problem?.operation, option.problem?.dataType]
    .filter(Boolean).join(' ').toLowerCase().includes(normalized)).slice(0, 50);
}

export function explorerCatalog(data, filters) {
  return selectBenchmarkCatalog(data, filters);
}

// Like canonical Overview, the range is commit-time based and anchored to published
// canonical history, not the viewer's clock or a branch execution.
export function benchmarkRangeData(data, range = 'ALL') {
  if (range === 'ALL' || !data.runs.length) return data;
  const days = historyRangeById.get(range)?.days;
  if (!days) return data;
  const latest = Math.max(...data.runs.map((run) => Date.parse(commitTimestampFor(run))));
  const anchor = new Date(latest);
  anchor.setUTCHours(0, 0, 0, 0);
  const cutoff = anchor.getTime() - (days - 1) * 86400000;
  return { ...data, runs: data.runs.filter((run) => Date.parse(commitTimestampFor(run)) >= cutoff) };
}
