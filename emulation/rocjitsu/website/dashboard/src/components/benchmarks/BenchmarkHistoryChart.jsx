import { useMemo, useState } from 'react';
import { Alert, useMediaQuery, useTheme } from '@mui/material';
import Chart from '../shared/Chart';
import ChartPointSelector from './ChartPointSelector';
import { benchmarkRangeData, benchmarkResultChoices } from './benchmarkExplorer';
import { selectBenchmarkSeries } from '../../data/selectors';
import { escapeHtml, formatDuration, formatFullDate, shortSha } from '../../utils/formatters';
import { chartLineStyle, chartPointStyle } from '../../utils/chartStyles';
import { durationAxisBounds } from '../../utils/durationAxis';
import { initialZoomWindow, zoomFromEvent, zoomIncludingIndexes, zoomIndexRange } from '../../utils/chartZoom';
import { categoryColor } from '../../theme/tokens';

const INITIAL_VISIBLE_RUNS = 45;

export default function BenchmarkHistoryChart({ data, filters, benchmark, range = 'ALL', height = 370, selectedRunIds = [], onSelectRecord, onOpenRecord, onSelectRun, showDetailsOnClick = false, showPointSelector = false, scrollZoomEnabled = true }) {
  const theme = useTheme();
  const compact = useMediaQuery(theme.breakpoints.down('sm'));
  const rangedData = useMemo(() => benchmarkRangeData(data, range), [data, range]);
  const viewModel = useMemo(() => selectBenchmarkSeries(rangedData, filters, benchmark.id), [benchmark.id, rangedData, filters]);
  const selectedRunIdSet = useMemo(() => new Set(selectedRunIds), [selectedRunIds]);
  const selectedIndexes = useMemo(() => viewModel.runs.map((run, index) => selectedRunIdSet.has(run.runId) ? index : -1).filter((index) => index >= 0), [selectedRunIdSet, viewModel.runs]);
  const [initialAnchorRunId] = useState(() => selectedRunIds.at(-1) ?? null);
  const initialAnchorIndex = viewModel.runs.findIndex((run) => run.runId === initialAnchorRunId);
  const [zoom, setZoom] = useState(() => zoomIncludingIndexes(initialZoomWindow(viewModel.runs.length, INITIAL_VISIBLE_RUNS, initialAnchorIndex), viewModel.runs.length, selectedIndexes));
  // Shared timeframe displays the entire chosen period; optional legacy scroll
  // zoom stays supported without forcing selected points outside that period.
  const displayZoom = scrollZoomEnabled ? zoomIncludingIndexes(zoom, viewModel.runs.length, selectedIndexes) : { start: 0, end: 100 };
  const { startIndex, endIndex } = zoomIndexRange(displayZoom, viewModel.runs.length);
  const values = viewModel.series.flatMap((series) => series.points.slice(startIndex, endIndex + 1).map((point) => point?.value).filter(Number.isFinite));
  const colorFor = (series) => categoryColor(theme.palette.mode, series.target, series.color);
  const option = {
    textStyle: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily },
    color: viewModel.series.map(colorFor),
    tooltip: {
      trigger: 'axis', triggerOn: showDetailsOnClick ? 'mousemove|click' : 'mousemove',
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
    ...(scrollZoomEnabled ? { dataZoom: [{ type: 'inside', ...displayZoom }] } : {}),
    series: viewModel.series.map((series) => ({
      name: series.name, type: 'line', data: series.points, smooth: false,
      showSymbol: true, symbol: 'circle', symbolSize: compact ? 4 : 5, connectNulls: false,
      lineStyle: { ...chartLineStyle(colorFor(series), 2), type: series.mode === 'MT' ? 'dashed' : 'solid' },
      itemStyle: chartPointStyle(colorFor(series), theme.palette.background.paper),
      emphasis: { focus: 'series', scale: 1.4 }, z: 3,
    })).concat(viewModel.series.map((series) => ({
      name: `${series.name} point details`, type: 'scatter', data: series.points,
      symbolSize: compact ? 18 : 14, itemStyle: { color: 'rgba(0,0,0,0.001)' },
      emphasis: { scale: false }, tooltip: { show: showDetailsOnClick }, z: 10,
    }))).concat(viewModel.series.map((series) => ({
      name: `${series.name} selected points`, type: 'scatter',
      data: series.points.map((point) => point?.record && selectedRunIdSet.has(point.record.run.runId) ? point : null),
      symbol: 'circle', symbolSize: compact ? 9 : 11,
      itemStyle: { ...chartPointStyle(colorFor(series), theme.palette.background.paper), borderWidth: 3 }, tooltip: { show: false }, z: 12,
    }))),
  };
  const choices = useMemo(() => showPointSelector ? benchmarkResultChoices(viewModel) : [], [showPointSelector, viewModel]);
  const [unavailableChoice, setUnavailableChoice] = useState(null);
  const events = {
    click: (event) => {
      if (event.componentType === 'series' && event.data?.record) onSelectRecord?.(event.data.record);
    },
    datazoom: (event) => setZoom((current) => zoomIncludingIndexes(zoomFromEvent(event, current), viewModel.runs.length, selectedIndexes)),
  };
  const helpId = `benchmark-chart-keyboard-help-${benchmark.id}`;
  return <>
    <Chart option={option} height={height} ariaLabel={`${benchmark.name} duration history`} ariaDescribedBy={showPointSelector ? helpId : undefined} onEvents={events} lazyUpdate={false} />
    {unavailableChoice && <Alert severity="info" onClose={() => setUnavailableChoice(null)} sx={{ mt: 1, overflowWrap: 'anywhere' }}>Unavailable · {unavailableChoice.target} · {unavailableChoice.mode} · {unavailableChoice.run.runId}. No published result for this workload and configuration in this attempt. This is not a failed or zero-duration measurement.</Alert>}
    {showPointSelector && <ChartPointSelector label={`${benchmark.name} result`} actionLabel={`Open ${benchmark.name} result`} description={`Every point can also be reached with the ${benchmark.name} result control below it.`} descriptionId={helpId} options={choices} onActivate={(choice) => {
      if (choice.record) { setUnavailableChoice(null); onOpenRecord?.(choice.record); }
      else { onSelectRun?.(choice.run.runId); setUnavailableChoice(choice); }
    }} />}
  </>;
}
