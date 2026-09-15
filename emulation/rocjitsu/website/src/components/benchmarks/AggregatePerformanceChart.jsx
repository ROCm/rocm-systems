import { useMemo, useState } from 'react';
import { Box, Chip, useTheme } from '@mui/material';
import Chart from '../shared/Chart';
import ChartPointSelector from './ChartPointSelector';
import { selectAggregateRunSeries } from '../../data/selectors';
import { commitTimestampFor } from '../../data/runOrdering';
import { escapeHtml, formatDuration, formatFullDate, shortSha } from '../../utils/formatters';
import { chartAreaGradient, chartLineStyle, chartPointStyle } from '../../utils/chartStyles';
import { chartGapPresentation } from '../../utils/chartGaps';
import {
  initialZoomWindow,
  zoomFromEvent,
  zoomIncludingIndexes,
  zoomIndexRange,
} from '../../utils/chartZoom';

const INITIAL_VISIBLE_RUNS = 45;

export default function AggregatePerformanceChart({
  data,
  filters,
  selectedRunIds,
  onSelectRun,
  onOpenRun,
  showDetailsOnClick = false,
  scrollZoomEnabled = true,
}) {
  const theme = useTheme();
  const viewModel = useMemo(() => selectAggregateRunSeries(data, filters), [data, filters]);
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
  const latestSelectedRunId = selectedRunIds.at(-1) ?? null;
  const latestSelectedRun = viewModel.runs.find((run) => run.runId === latestSelectedRunId) ?? null;
  const visibleChartValues = viewModel.series.flatMap((series) => (
    series.data
      .slice(startIndex, endIndex + 1)
      .map((point) => point.value)
      .filter(Number.isFinite)
  ));
  const seriesPresentations = viewModel.series.map((series) => ({
    ...series,
    ...chartGapPresentation(series.data),
  }));
  const minimumValue = visibleChartValues.length ? Math.min(...visibleChartValues) : 0;
  const incompleteMarkerData = seriesPresentations.flatMap((series) => (
    series.data.flatMap((record, index) => {
      if (Number.isFinite(record.value) || record.total === 0) return [];
      const estimatedValue = series.estimatedValues[index];
      const value = Number.isFinite(estimatedValue) ? estimatedValue : minimumValue;
      const selectedTests = (record.run.tests ?? []).filter((test) => (
        test.target === series.target && filters.suites.includes(test.suite)
      ));
      const hasFailure = selectedTests.some((test) => test.status === 'failed');
      const statusColor = hasFailure ? theme.palette.error.main : theme.palette.warning.main;
      const selected = selectedRunIdSet.has(record.run.runId);
      return [{
        ...record,
        value: [index, value],
        incomplete: true,
        selected,
        symbolSize: selected ? 17 : 14,
        itemStyle: {
          ...chartPointStyle(statusColor, theme.palette.background.paper),
          borderWidth: selected ? 4 : 2,
          shadowBlur: selected ? 13 : 9,
          shadowColor: statusColor,
        },
        label: {
          show: true,
          formatter: `${record.completed}/${record.total}`,
          position: 'top',
          color: statusColor,
          fontSize: 10,
          fontWeight: 700,
        },
      }];
    })
  ));
  const selectedMarkerData = viewModel.series.flatMap((series) => (
    series.data.flatMap((record, index) => (
      selectedRunIdSet.has(record.run.runId) && Number.isFinite(record.value)
        ? [{
            ...record,
            value: [index, record.value],
            itemStyle: { ...chartPointStyle(series.color, theme.palette.background.paper), borderWidth: 3 },
          }]
        : []
    ))
  ));
  selectedIndexes.forEach((index) => {
    const unavailableTargets = viewModel.series
      .filter((series) => (
        !Number.isFinite(series.data[index]?.value) && series.data[index]?.total === 0
      ))
      .map((series) => series.target);
    if (unavailableTargets.length === 0) return;
    selectedMarkerData.push({
      value: [index, minimumValue],
      run: viewModel.runs[index],
      unavailableTargets,
      itemStyle: {
        ...chartPointStyle(theme.palette.text.disabled, theme.palette.background.paper),
        borderWidth: 3,
      },
      label: {
        show: true,
        formatter: `N/A · ${unavailableTargets.join(', ')}`,
        position: 'top',
        color: theme.palette.text.secondary,
        fontSize: 10,
        fontWeight: 700,
      },
    });
  });

  const option = {
    tooltip: {
      trigger: 'axis',
      triggerOn: showDetailsOnClick ? 'mousemove|click' : 'mousemove',
      axisPointer: { type: 'line', lineStyle: { color: theme.palette.text.disabled, type: 'dashed' } },
      backgroundColor: theme.palette.background.paper,
      borderColor: theme.palette.divider,
      textStyle: { color: theme.palette.text.primary },
      formatter: (parameters) => {
        const allPoints = Array.isArray(parameters) ? parameters : [parameters];
        const point = allPoints.find((parameter) => parameter.data?.run);
        if (!point) return '';
        const run = point.data.run;
        const usable = allPoints.filter((parameter) => (
          parameter.seriesType === 'line' && Number.isFinite(parameter.data?.value)
        ));
        const selectedTests = (run.tests ?? []).filter((test) => (
          filters.targets.includes(test.target) && filters.suites.includes(test.suite)
        ));
        const totalTestCount = selectedTests.length;
        const unavailable = allPoints.find((parameter) => (
          parameter.seriesName === 'Selected run' && parameter.data?.unavailableTargets
        ))?.data.unavailableTargets;
        const incomplete = allPoints.find((parameter) => (
          parameter.seriesName === 'Incomplete aggregate results' && parameter.data?.incomplete
        ));
        return [
          `<strong>Commit ${escapeHtml(shortSha(run))}</strong>`,
          `Commit time · ${escapeHtml(formatFullDate(commitTimestampFor(run)))}`,
          `Execution time · ${escapeHtml(formatFullDate(run.timestamp))}`,
          `Run type · ${run.trigger === 'manual' ? 'Manual' : 'Auto'}`,
          ...usable.map((targetPoint) => (
            `${targetPoint.marker}${escapeHtml(targetPoint.seriesName)}&nbsp;&nbsp;<strong>${formatDuration(targetPoint.data.value)}</strong>`
          )),
          incomplete
            ? `${incomplete.marker}${escapeHtml(incomplete.data.target)}&nbsp;&nbsp;<strong>${incomplete.data.completed}/${incomplete.data.total} completed</strong>`
            : null,
          unavailable?.length ? `Unavailable targets · ${escapeHtml(unavailable.join(', '))}` : null,
          `Selected results · ${selectedTests.filter((test) => test.status === 'completed').length}/${totalTestCount} completed`,
        ].filter(Boolean).join('<br/>');
      },
    },
    legend: {
      data: viewModel.series.map((series) => series.target),
      top: 0,
      right: 0,
      textStyle: { color: theme.palette.text.secondary },
    },
    grid: { left: 58, right: 22, top: 42, bottom: 82 },
    xAxis: {
      type: 'category',
      name: 'Commit Date (UTC)',
      nameLocation: 'middle',
      nameGap: 48,
      boundaryGap: false,
      data: viewModel.labels,
      axisTick: { show: false },
      axisLabel: { color: theme.palette.text.secondary, fontSize: 10, lineHeight: 14, hideOverlap: true },
      nameTextStyle: { color: theme.palette.text.secondary, fontSize: 11 },
      axisLine: { lineStyle: { color: theme.palette.divider } },
    },
    yAxis: {
      type: 'value',
      name: 'Seconds',
      scale: true,
      nameTextStyle: { color: theme.palette.text.secondary, align: 'right' },
      axisLabel: { color: theme.palette.text.secondary },
      splitLine: { lineStyle: { color: theme.palette.divider, type: 'dashed' } },
    },
    dataZoom: [
      {
        type: 'inside',
        disabled: !scrollZoomEnabled,
        ...displayZoom,
        zoomOnMouseWheel: true,
        moveOnMouseWheel: true,
        moveOnMouseMove: true,
      },
      {
        type: 'slider',
        ...displayZoom,
        height: 18,
        bottom: 8,
        borderColor: theme.palette.divider,
        backgroundColor: theme.palette.action.hover,
        fillerColor: theme.palette.action.selected,
        dataBackground: { lineStyle: { color: theme.palette.primary.main }, areaStyle: { color: theme.palette.primary.main, opacity: 0.1 } },
        selectedDataBackground: { lineStyle: { color: theme.palette.primary.main }, areaStyle: { color: theme.palette.primary.main, opacity: 0.18 } },
        textStyle: { color: theme.palette.text.secondary, fontSize: 9 },
      },
    ],
    series: [
      ...seriesPresentations.map((series) => ({
        name: series.target,
        type: 'line',
        data: series.data,
        showSymbol: false,
        symbol: 'circle',
        symbolSize: 7,
        connectNulls: false,
        smooth: 0.12,
        lineStyle: chartLineStyle(series.color, 2.6),
        itemStyle: chartPointStyle(series.color, theme.palette.background.paper),
        emphasis: { focus: 'series', scale: 1.6, lineStyle: { width: 3.2 } },
        areaStyle: {
          color: chartAreaGradient(series.color, viewModel.series.length === 1 ? 0.24 : 0.1),
          opacity: 1,
        },
        z: 3,
      })),
      ...seriesPresentations.flatMap((series) => (
        series.segments.map((segment, index) => ({
          name: `${series.target} missed-data bridge ${index + 1}`,
          type: 'line',
          data: segment,
          showSymbol: false,
          symbol: 'none',
          connectNulls: false,
          smooth: 0.12,
          silent: true,
          tooltip: { show: false },
          lineStyle: { ...chartLineStyle(series.color, 2.3), type: 'dotted', opacity: 0.85 },
          emphasis: { disabled: true },
          z: 4,
        }))
      )),
      ...seriesPresentations.map((series) => ({
        name: `${series.target} point selection`,
        type: 'scatter',
        data: series.data,
        symbolSize: 18,
        itemStyle: { color: 'rgba(0, 0, 0, 0.001)' },
        emphasis: { scale: false },
        tooltip: { show: showDetailsOnClick },
        z: 10,
      })),
      ...(incompleteMarkerData.length > 0 ? [{
        name: 'Incomplete aggregate results',
        type: 'scatter',
        data: incompleteMarkerData,
        symbol: 'diamond',
        symbolSize: 14,
        clip: false,
        emphasis: { scale: 1.3 },
        tooltip: { show: true },
        z: 11,
      }] : []),
      {
        name: 'Selected run',
        type: 'scatter',
        data: selectedMarkerData,
        symbol: 'circle',
        symbolSize: 13,
        clip: false,
        label: { show: false },
        z: 12,
      },
    ],
  };
  const runOptions = useMemo(() => [...viewModel.runs].reverse().map((run) => ({
    id: run.runId,
    label: `${shortSha(run)} · ${formatFullDate(run.timestamp)}`,
    run,
  })), [viewModel.runs]);
  const chartEvents = useMemo(() => ({
    click: (parameters) => {
      if (parameters.componentType === 'series' && parameters.data?.run) {
        setZoom((current) => zoomIncludingIndexes(
          current,
          viewModel.runs.length,
          selectedIndexes,
        ));
        onSelectRun(parameters.data.run);
      }
    },
    datazoom: (parameters) => {
      setZoom((current) => zoomIncludingIndexes(
        zoomFromEvent(parameters, current),
        viewModel.runs.length,
        selectedIndexes,
      ));
    },
  }), [onSelectRun, selectedIndexes, viewModel.runs.length]);

  return (
    <Box data-testid="aggregate-performance-chart">
      {latestSelectedRun && (
        <Box sx={{ display: 'flex', justifyContent: { xs: 'flex-start', sm: 'flex-end' }, mb: 1 }}>
          <Chip
            size="small"
            color="primary"
            variant="outlined"
            label={selectedRunIds.length === 1
              ? `Selected ${shortSha(latestSelectedRun)} · ${latestSelectedRun.trigger === 'manual' ? 'Manual' : 'Auto'}`
              : `${selectedRunIds.length} selected runs · latest selection ${shortSha(latestSelectedRun)}`}
          />
        </Box>
      )}
      <Chart
        option={option}
        height={430}
        ariaLabel="Aggregate duration history for all runs"
        ariaDescribedBy="aggregate-chart-keyboard-help"
        onEvents={chartEvents}
      />
      <ChartPointSelector
        label="Aggregate run"
        actionLabel="Open run details"
        description="Every point in this chart can also be reached with the Aggregate run control below it."
        descriptionId="aggregate-chart-keyboard-help"
        options={runOptions}
        onActivate={(choice) => onOpenRun(choice.run)}
      />
    </Box>
  );
}
