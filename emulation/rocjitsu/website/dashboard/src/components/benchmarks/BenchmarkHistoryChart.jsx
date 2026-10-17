import { useMemo } from 'react';
import { useMediaQuery, useTheme } from '@mui/material';
import Chart from '../shared/Chart';
import { benchmarkRangeData } from './benchmarkExplorer';
import { selectBenchmarkSeries } from '../../data/selectors';
import { escapeHtml, formatDuration, formatFullDate, shortSha } from '../../utils/formatters';
import { chartLineStyle, chartPointStyle } from '../../utils/chartStyles';
import { durationAxisBounds } from '../../utils/durationAxis';
import { categoryColor } from '../../theme/tokens';

export default function BenchmarkHistoryChart({ data, filters, benchmark, range = 'ALL', height = 370, selectedRunIds = [], onSelectRecord }) {
  const theme = useTheme();
  const compact = useMediaQuery(theme.breakpoints.down('sm'));
  const rangedData = useMemo(() => benchmarkRangeData(data, range), [data, range]);
  const viewModel = useMemo(() => selectBenchmarkSeries(rangedData, filters, benchmark.id), [benchmark.id, rangedData, filters]);
  const selectedRunIdSet = useMemo(() => new Set(selectedRunIds), [selectedRunIds]);
  const values = viewModel.series.flatMap((series) => series.points.map((point) => point?.value).filter(Number.isFinite));
  const colorFor = (series) => categoryColor(theme.palette.mode, series.target, series.color);
  const option = {
    textStyle: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily },
    color: viewModel.series.map(colorFor),
    tooltip: {
      trigger: 'axis', triggerOn: 'mousemove|click',
      confine: true, padding: [6, 8],
      backgroundColor: theme.palette.background.paper, borderColor: theme.palette.divider,
      textStyle: { color: theme.palette.text.primary, fontFamily: theme.typography.fontFamily, fontSize: 11, lineHeight: 16 },
      extraCssText: 'max-width:240px;white-space:normal;overflow-wrap:anywhere;pointer-events:none;',
      axisPointer: { type: 'line', snap: true, lineStyle: { color: theme.palette.text.disabled, type: 'dotted' } },
      formatter: (parameters) => {
        const points = (Array.isArray(parameters) ? parameters : [parameters]).filter((point) => point.data?.record && point.seriesType === 'line');
        if (!points.length) return 'No measured result for this benchmark';
        const run = points[0].data.record.run;
        return [
          `<strong>Commit ${escapeHtml(shortSha(run))}</strong>`,
          escapeHtml(formatFullDate(run.timestamp)),
          ...points.map((point) => `${point.marker}${escapeHtml(point.seriesName)}&nbsp;&nbsp;<strong>${formatDuration(point.data.record.test.durationSeconds)}</strong>`),
        ].join('<br/>');
      },
    },
    grid: { left: 46, right: 14, top: 24, bottom: 45 },
    xAxis: {
      type: 'category', data: viewModel.labels,
      axisLabel: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily, fontSize: 11, lineHeight: 15, hideOverlap: true },
      axisTick: { show: false }, axisLine: { lineStyle: { color: theme.palette.divider } },
    },
    yAxis: {
      type: 'value', name: 'Seconds', ...durationAxisBounds(values),
      nameTextStyle: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily, fontSize: 11 },
      axisLabel: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily, fontSize: 11 },
      splitLine: { lineStyle: { color: theme.palette.divider, type: 'dotted' } },
    },
    series: viewModel.series.map((series) => ({
      name: series.name, type: 'line', data: series.points, smooth: false,
      showSymbol: true, symbol: 'circle', symbolSize: compact ? 4 : 5, connectNulls: false,
      lineStyle: { ...chartLineStyle(colorFor(series), 2), type: series.mode === 'MT' ? 'dashed' : 'solid' },
      itemStyle: chartPointStyle(colorFor(series), theme.palette.background.paper),
      emphasis: { focus: 'series', scale: 1.4 }, z: 3,
    })).concat(viewModel.series.map((series) => ({
      name: `${series.name} point details`, type: 'scatter', data: series.points,
      symbolSize: compact ? 18 : 14, itemStyle: { color: 'rgba(0,0,0,0.001)' },
      emphasis: { scale: false }, tooltip: { show: true }, z: 10,
    }))).concat(viewModel.series.map((series) => ({
      name: `${series.name} selected points`, type: 'scatter',
      data: series.points.map((point) => point?.record && selectedRunIdSet.has(point.record.run.runId) ? point : null),
      symbol: 'circle', symbolSize: compact ? 9 : 11,
      itemStyle: { ...chartPointStyle(colorFor(series), theme.palette.background.paper), borderWidth: 3 }, tooltip: { show: false }, z: 12,
    }))),
  };
  const events = {
    click: (event) => {
      if (event.componentType === 'series' && event.data?.record) onSelectRecord?.(event.data.record);
    },
  };
  return (
    <Chart option={option} height={height} ariaLabel={`${benchmark.name} duration history`} onEvents={events} lazyUpdate={false} />
  );
}
