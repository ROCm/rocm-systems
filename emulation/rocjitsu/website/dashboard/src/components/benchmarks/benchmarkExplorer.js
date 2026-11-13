import { formatDuration, formatFullDate, shortSha } from '../../utils/formatters';
import { commitTimestampFor, hasCommitTimestamp } from '../../data/runOrdering';
import { resolveHistoryRange } from '../../config/historyRanges';
import { selectDaySlots } from '../../data/overviewProjection';
import { MAX_BENCHMARK_CHARTS } from '../../config/benchmarkGrid';

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

// The runs the history charts actually plot for a range: benchmarkRangeData narrowed by the same
// day-slot policy as the chart (the last run per commit day, except 1W). The chart, Inspect results
// and default workload selection all read this one run set, so every offered result can be highlighted.
export function benchmarkPlottedData(data, range = 'ALL') {
  const ranged = benchmarkRangeData(data, range);
  const { slots } = selectDaySlots(ranged.runs, data.generatedAt, range);
  return { ...ranged, runs: slots.flatMap(({ run }) => run ? [run] : []) };
}

// Local presentation helpers for Benchmark Explorer.
export function boundedGridSelection(options, maximum = MAX_BENCHMARK_CHARTS) {
  return [...new Map(options.map((option) => [option.id, option])).values()].slice(0, Math.min(maximum, MAX_BENCHMARK_CHARTS));
}

export function filterBenchmarkOptions(options, query) {
  const normalized = query.trim().toLowerCase();
  return options.filter((option) => [option.id, option.name, option.suite, option.operation, option.problem?.dtype]
    .filter((value) => value != null).join(' ').toLowerCase().includes(normalized)).slice(0, 50);
}

// Like canonical Overview, the range is commit-time based and anchored to published
// canonical history, not the viewer's clock or a branch execution.
export function benchmarkRangeData(data, range = 'ALL') {
  const { days } = resolveHistoryRange(range);
  if (!days || !data.runs.length) return data;
  const latest = Math.max(...data.runs.filter(hasCommitTimestamp).map((run) => Date.parse(commitTimestampFor(run))));
  if (!Number.isFinite(latest)) return { ...data, runs: [] };
  const anchor = new Date(latest);
  anchor.setUTCHours(0, 0, 0, 0);
  const cutoff = anchor.getTime() - (days - 1) * 86400000;
  return { ...data, runs: data.runs.filter((run) => Date.parse(commitTimestampFor(run)) >= cutoff) };
}
