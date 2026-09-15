import { useMemo, useState } from 'react';
import { useMediaQuery, useTheme } from '@mui/material';
import Chart from '../shared/Chart';
import ChartPointSelector from './ChartPointSelector';
import { selectBenchmarkSeries } from '../../data/selectors';
import { commitTimestampFor } from '../../data/runOrdering';
import { escapeHtml, formatDuration, formatFullDate, shortSha } from '../../utils/formatters';
import { chartAreaGradient, chartLineStyle, chartPointStyle } from '../../utils/chartStyles';
import { chartGapPresentation } from '../../utils/chartGaps';
import { durationAxisBounds } from '../../utils/durationAxis';
import { initialZoomWindow, zoomFromEvent, zoomIncludingIndexes, zoomIndexRange } from '../../utils/chartZoom';

const INITIAL_VISIBLE_RUNS = 45;

export default function BenchmarkHistoryChart({
  data,
  filters,
  benchmark,
  height = 370,
  selectedRunIds = [],
  onSelectRecord,
  onOpenRecord,
  onSelectRun,
  showDetailsOnClick = false,
  showPointSelector = false,
  scrollZoomEnabled = true,
}) {
  const theme = useTheme();
  const compact = useMediaQuery(theme.breakpoints.down('sm'));
  const viewModel = useMemo(
    () => selectBenchmarkSeries(data, filters, benchmark.id),
    [benchmark.id, data, filters],
  );
  const selectedRunIdSet = useMemo(() => new Set(selectedRunIds), [selectedRunIds]);
  const selectedIndexes = useMemo(() => viewModel.runs
    .map((run, index) => (selectedRunIdSet.has(run.runId) ? index : -1))
    .filter((index) => index >= 0), [selectedRunIdSet, viewModel.runs]);
  const [initialAnchorRunId] = useState(() => selectedRunIds.at(-1) ?? null);
  const initialAnchorIndex = viewModel.runs.findIndex((run) => run.runId === initialAnchorRunId);
  const [zoom, setZoom] = useState(() => (
    zoomIncludingIndexes(
      initialZoomWindow(viewModel.runs.length, INITIAL_VISIBLE_RUNS, initialAnchorIndex),
      viewModel.runs.length,
      selectedIndexes,
    )
  ));
  const displayZoom = zoomIncludingIndexes(zoom, viewModel.runs.length, selectedIndexes);
  const { startIndex, endIndex } = zoomIndexRange(displayZoom, viewModel.runs.length);
  const chartValues = viewModel.series.flatMap((series) => (
    series.points
      .slice(startIndex, endIndex + 1)
      .map((point) => point?.value)
      .filter(Number.isFinite)
  ));
  const durationAxis = durationAxisBounds(chartValues);
  const seriesPresentations = viewModel.series.map((series) => ({
    ...series,
    ...chartGapPresentation(series.points),
  }));
  const minimumValue = chartValues.length ? Math.min(...chartValues) : 0;
  const statusMarkerData = seriesPresentations.flatMap((series) => (
    series.records.flatMap((record, index) => {
      if (!record || record.test.status === 'completed') return [];
      const value = series.estimatedValues[index];
      if (!Number.isFinite(value)) return [];
      const statusColor = record.test.status === 'failed'
        ? theme.palette.error.main
        : theme.palette.warning.main;
      const selected = selectedRunIdSet.has(record.run.runId);
      return [{
        value: [index, value],
        record,
        target: series.name,
        statusPoint: true,
        selected,
        symbolSize: selected ? (compact ? 14 : 17) : (compact ? 11 : 14),
        itemStyle: {
          ...chartPointStyle(statusColor, theme.palette.background.paper),
          borderWidth: selected ? 4 : 2,
          shadowBlur: selected ? 13 : 9,
          shadowColor: statusColor,
        },
        label: {
          show: true,
          formatter: record.test.status === 'failed' ? 'Failed' : 'Timeout',
          position: 'top',
          color: statusColor,
          fontSize: 10,
          fontWeight: 700,
        },
      }];
    })
  ));
  const unavailableMarkerData = selectedIndexes.flatMap((index) => {
    const unavailableTargets = viewModel.series
      .filter((series) => !series.records[index])
      .map((series) => series.name);
    if (unavailableTargets.length === 0) return [];
    return [{
      value: [index, minimumValue],
      run: viewModel.runs[index],
      unavailableTargets,
      label: {
        show: true,
        formatter: unavailableTargets.length === 1 ? 'N/A' : `N/A · ${unavailableTargets.join(', ')}`,
        position: 'top',
        color: theme.palette.text.secondary,
        fontSize: 10,
        fontWeight: 700,
      },
    }];
  });
  const option = {
    color: viewModel.series.map((series) => series.color),
    tooltip: {
      trigger: 'axis',
      triggerOn: showDetailsOnClick ? 'mousemove|click' : 'mousemove',
      backgroundColor: theme.palette.background.paper,
      borderColor: theme.palette.divider,
      textStyle: { color: theme.palette.text.primary },
      formatter: (parameters) => {
        const points = (Array.isArray(parameters) ? parameters : [parameters]).filter((point) => (
          point.data?.record && (point.seriesType === 'line' || point.data.statusPoint)
        ));
        if (points.length === 0) return 'No result for this benchmark';
        const run = points[0].data.record.run;
        return [
          `<strong>Run time · ${escapeHtml(formatFullDate(run.timestamp))}</strong>`,
          `Commit ${escapeHtml(shortSha(run))} · ${escapeHtml(formatFullDate(commitTimestampFor(run)))}`,
          ...points.map((point) => {
            const { test } = point.data.record;
            const target = escapeHtml(point.data.target ?? point.seriesName);
            return test.status === 'completed'
              ? `${point.marker}${target}&nbsp;&nbsp;<strong>${formatDuration(test.durationSeconds)}</strong> · Completed`
              : `${point.marker}${target}&nbsp;&nbsp;<strong>${test.status === 'failed' ? 'Failed' : 'Timeout'}</strong>`;
          }),
        ].join('<br/>');
      },
    },
    legend: {
      data: viewModel.series.map((series) => series.name),
      top: 0,
      right: 0,
      textStyle: { color: theme.palette.text.secondary },
    },
    grid: { left: 52, right: 18, top: 42, bottom: 58 },
    xAxis: {
      type: 'category',
      name: 'Commit date / commit',
      nameLocation: 'middle',
      nameGap: 43,
      data: viewModel.labels,
      axisLabel: { color: theme.palette.text.secondary, fontSize: 10, lineHeight: 14, hideOverlap: true },
      nameTextStyle: { color: theme.palette.text.secondary, fontSize: 11 },
      axisLine: { lineStyle: { color: theme.palette.divider } },
    },
    yAxis: {
      type: 'value',
      name: 'Seconds',
      ...durationAxis,
      axisLabel: { color: theme.palette.text.secondary },
      splitLine: { lineStyle: { color: theme.palette.divider, type: 'dashed' } },
    },
    dataZoom: [{ type: 'inside', disabled: !scrollZoomEnabled, ...displayZoom }],
    series: seriesPresentations.map((series) => ({
        name: series.name,
        type: 'line',
        data: series.points,
        smooth: 0.18,
        showSymbol: false,
        symbol: 'circle',
        symbolSize: compact ? 4 : 6,
        connectNulls: false,
        lineStyle: chartLineStyle(series.color, 2.7),
        itemStyle: chartPointStyle(series.color, theme.palette.background.paper),
        areaStyle: { color: chartAreaGradient(series.color, 0.13), opacity: 1 },
        emphasis: { focus: 'series', scale: 1.6, lineStyle: { width: 3.3 } },
        z: 3,
      })).concat(seriesPresentations.flatMap((series) => (
      series.segments.map((segment, index) => ({
        name: `${series.name} missed-data bridge ${index + 1}`,
        type: 'line',
        data: segment,
        smooth: 0.18,
        showSymbol: false,
        symbol: 'none',
        connectNulls: false,
        silent: true,
        tooltip: { show: false },
        lineStyle: { ...chartLineStyle(series.color, 2.4), type: 'dotted', opacity: 0.85 },
        emphasis: { disabled: true },
        z: 4,
      }))
    ))).concat(seriesPresentations.map((series) => ({
      name: `${series.name} point details`,
      type: 'scatter',
      data: series.points,
      symbolSize: compact ? 18 : 14,
      itemStyle: { color: 'rgba(0, 0, 0, 0.001)' },
      emphasis: { scale: false },
      tooltip: { show: showDetailsOnClick },
      z: 10,
    }))).concat(statusMarkerData.length > 0 ? [{
      name: 'Failed or timed-out test results',
      type: 'scatter',
      data: statusMarkerData,
      symbol: 'diamond',
      symbolSize: compact ? 11 : 14,
      clip: false,
      emphasis: { scale: 1.3 },
      tooltip: { show: true },
      z: 11,
    }] : []).concat(seriesPresentations.map((series) => ({
      name: `${series.name} selected points`,
      type: 'scatter',
      data: series.points.map((point) => (
        point?.record && selectedRunIdSet.has(point.record.run.runId) ? point : null
      )),
      symbol: 'circle',
      symbolSize: compact ? 9 : 12,
      clip: false,
      itemStyle: { ...chartPointStyle(series.color, theme.palette.background.paper), borderWidth: 3 },
      emphasis: { scale: 1.25 },
      tooltip: { show: false },
      z: 12,
    }))).concat(unavailableMarkerData.length > 0 ? [{
      name: 'Selected runs unavailable',
      type: 'scatter',
      data: unavailableMarkerData,
      symbol: 'circle',
      symbolSize: compact ? 9 : 12,
      clip: false,
      itemStyle: { ...chartPointStyle(theme.palette.text.disabled, theme.palette.background.paper), borderWidth: 3 },
      emphasis: { scale: 1.25 },
      tooltip: { show: false },
      z: 12,
    }] : []),
  };
  const recordOptions = useMemo(() => (showPointSelector ? viewModel.series
    .flatMap((series) => series.records.map((record, index) => ({ record, index })))
    .filter(({ record }) => record)
    .sort((left, right) => right.index - left.index)
    .map(({ record }) => ({
      id: `${record.test.target}:${record.run.runId}`,
      label: `${record.test.target} · ${shortSha(record.run)} · ${record.test.status === 'completed'
        ? formatDuration(record.test.durationSeconds)
        : record.test.status}`,
      record,
    })) : []), [showPointSelector, viewModel.series]);
  const chartEvents = useMemo(() => ({
    click: (parameters) => {
      if (parameters.componentType === 'series' && parameters.data?.record) {
        setZoom((current) => zoomIncludingIndexes(
          current,
          viewModel.runs.length,
          selectedIndexes,
        ));
        onSelectRecord(parameters.data.record);
      } else if (parameters.componentType === 'series' && parameters.data?.run) {
        setZoom((current) => zoomIncludingIndexes(
          current,
          viewModel.runs.length,
          selectedIndexes,
        ));
        onSelectRun(parameters.data.run.runId);
      }
    },
    datazoom: (parameters) => {
      setZoom((current) => zoomIncludingIndexes(
        zoomFromEvent(parameters, current),
        viewModel.runs.length,
        selectedIndexes,
      ));
    },
  }), [onSelectRecord, onSelectRun, selectedIndexes, viewModel.runs.length]);

  const keyboardHelpId = `benchmark-chart-keyboard-help-${benchmark.id}`;

  return (
    <>
      <Chart
        option={option}
        height={height}
        ariaLabel={`${benchmark.name} duration history`}
        ariaDescribedBy={showPointSelector ? keyboardHelpId : undefined}
        onEvents={chartEvents}
      />
      {showPointSelector && (
        <ChartPointSelector
          label={`${benchmark.name} result`}
          actionLabel={`Open ${benchmark.name} result`}
          description={`Every point in this chart can also be reached with the ${benchmark.name} result control below it.`}
          descriptionId={keyboardHelpId}
          options={recordOptions}
          onActivate={(choice) => onOpenRecord(choice.record)}
        />
      )}
    </>
  );
}
