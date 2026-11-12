import { fontSizes } from '../../theme/typography';
import { useMemo } from 'react';
import useElementWidth from '../../hooks/useElementWidth';
import { useMediaQuery, useTheme } from '@mui/material';
import { alpha } from '@mui/material/styles';
import Chart from '../shared/Chart';
import { selectBenchmarkSeries } from '../../data/selectors';
import { selectDaySlots } from '../../data/overviewProjection';
import { commitTimestampFor } from '../../data/runOrdering';
import { dayAxisBounds, dayXAxis } from '../../utils/dayAxis';
import { escapeHtml, formatDuration, formatFullDate, shortSha } from '../../utils/formatters';
import { CHART_GRID_INSETS, chartLineStyle, chartPointStyle, chartPointSize, chartSwatch } from '../../utils/chartStyles';
import { durationAxisBounds } from '../../utils/durationAxis';
import { targetColor } from '../../theme/targetColors';

// Cards are ~300px wide, so their dots may sit closer than on the full-width trend chart, but still sparse.
const MIN_CARD_DOT_SPACING = 44;
const LEFT_PAD_FRACTION = 0.08; // of the visible span

export default function BenchmarkHistoryChart({ data, filters, benchmark, range = 'ALL', height = 370, selectedRunIds = [], onSelectRecord }) {
  const theme = useTheme();
  const [chartRef, chartWidth] = useElementWidth();
  const compact = useMediaQuery(theme.breakpoints.down('sm'));
  // Same commit-day layout and axis as Overview's performance trend: one run per day
  // (every run in 1W), x = commit time; the line connects across days without a measured result.
  const model = useMemo(() => {
    const view = selectBenchmarkSeries(data, filters, benchmark.id);
    const layout = selectDaySlots(data.runs, data.generatedAt, range);
    const series = view.series.map((row) => {
      const byRun = new Map(view.runs.map((run, index) => [run, row.points[index]]));
      return { ...row, points: layout.slots.map((slot) => {
        const point = slot.run ? byRun.get(slot.run) : null;
        return point ? { value: [slot.x, point.value], record: point.record } : { value: [slot.x, null] };
      }) };
    });
    series.forEach((row) => { row.measured = row.points.map((point) => ({ value: point.value[1] })); });
    const firstData = layout.slots.findIndex((slot, index) => slot.run && series.some((row) => row.points[index].record));
    return { series, layout, slots: layout.slots, bounds: dayAxisBounds(layout, firstData < 0 ? null : firstData) };
  }, [data, filters, benchmark.id, range]);
  const viewModel = model;
  const selectedRunIdSet = useMemo(() => new Set(selectedRunIds), [selectedRunIds]);
  const values = viewModel.series.flatMap((series) => series.points.map((point) => point.value[1]).filter(Number.isFinite));
  const colorFor = (series) => targetColor(theme, series.target);
  const option = {
    color: viewModel.series.map(colorFor),
    tooltip: {
      trigger: 'axis', triggerOn: 'mousemove|click', className: 'performance-trend-tooltip',
      axisPointer: { type: 'line', snap: true, lineStyle: { color: theme.palette.text.secondary, type: 'dotted' } },
      formatter: (parameters) => {
        const points = (Array.isArray(parameters) ? parameters : [parameters]).filter((point) => point.data?.record && point.seriesType === 'line');
        if (!points.length) return 'No measured result for this benchmark';
        const run = points[0].data.record.run;
        return [
          `<div><b>Commit ${escapeHtml(shortSha(run))}</b></div>`,
          `<div>${escapeHtml(formatFullDate(commitTimestampFor(run)))}</div>`,
          // Hollow "single" points have a paper-colored fill, so the swatch uses the series line color.
          ...points.map((point) => `<div class="performance-trend-tooltip-row">${chartSwatch(escapeHtml(colorFor(viewModel.series[point.seriesIndex])), point.data.record.test.mode)}${escapeHtml(point.seriesName)} · <b>${formatDuration(point.data.record.test.durationSeconds)}</b></div>`),
        ].join('');
      },
    },
    grid: { ...CHART_GRID_INSETS, top: 24, bottom: 30 },
    // Extra room on the left so the first point clears the y axis.
    xAxis: { ...dayXAxis(viewModel.bounds, theme, { leftPad: (viewModel.bounds.lastX - viewModel.bounds.firstX) * LEFT_PAD_FRACTION }), name: 'Date (UTC)', nameLocation: 'middle', nameGap: 18 },
    yAxis: {
      type: 'value', name: 'Seconds', ...durationAxisBounds(values),
      nameTextStyle: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily, fontSize: fontSizes.caption },
      axisLabel: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily, fontSize: fontSizes.caption },
      axisLine: { show: true, lineStyle: { width: 1, color: theme.palette.dashboard.border } },
      splitLine: { lineStyle: { color: alpha(theme.palette.dashboard.border, 0.55), width: 1 } },
    },
    series: viewModel.series.map((series) => ({
      name: series.name, type: 'line', data: series.points, smooth: false,
      showSymbol: true, symbol: series.mode === 'single' ? 'diamond' : 'circle', symbolSize: (_value, params) => chartPointSize(series.measured, params.dataIndex, series.mode, 'value', chartWidth, MIN_CARD_DOT_SPACING), connectNulls: true,
      lineStyle: chartLineStyle(colorFor(series), series.mode),
      itemStyle: chartPointStyle(colorFor(series), theme.palette.background.paper, series.mode),
      emphasis: { focus: 'series', scale: 1.4 }, z: 3,
    })).concat(viewModel.series.map((series) => ({
      name: `${series.name} point details`, type: 'scatter', data: series.points,
      // Keep the hit area painted for chart input without a visible extra point.
      symbolSize: compact ? 18 : 14, itemStyle: { color: alpha(theme.palette.primary.main, 0.001) },
      emphasis: { scale: false }, tooltip: { show: true }, z: 10,
    }))).concat(viewModel.series.map((series) => ({
      name: `${series.name} selected points`, type: 'scatter',
      data: series.points.map((point) => point?.record && selectedRunIdSet.has(point.record.run.runId) ? point : null),
      symbol: 'circle', symbolSize: compact ? 9 : 11,
      itemStyle: { ...chartPointStyle(colorFor(series), theme.palette.background.paper, series.mode), borderWidth: 3 }, tooltip: { show: false }, z: 12,
    }))),
  };
  const events = {
    click: (event) => {
      if (event.componentType === 'series' && event.data?.record) onSelectRecord?.(event.data.record);
    },
  };
  return (
    <div ref={chartRef}><Chart option={option} height={height} ariaLabel={`${benchmark.name} duration history`} onEvents={events} lazyUpdate={false} /></div>
  );
}
