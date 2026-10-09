import { memo, useCallback, useId, useMemo, useRef, useState } from 'react';
import { Box, Link, Stack, ToggleButton, ToggleButtonGroup, Typography, useTheme } from '@mui/material';
import Chart from '../shared/Chart';
import SectionCard from '../shared/SectionCard';
import { escapeHtml, formatDuration, formatFullDate, formatPercent, formatShortDate, shortSha } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';
import { chartAreaGradient, chartPointStyle } from '../../utils/chartStyles';
import { commitTimestampFor } from '../../data/runOrdering';
import { historyRanges, historyRangeById } from '../../config/historyRanges';
import { colorTokens, monoFont } from '../../theme/tokens';
import { visuallyHiddenStyles } from '../../theme/styles';
import { trendKeyIndex, nearestTrendIndex, shouldHideTrendPointer } from './overviewPresentation';

// Inspection text can rerender without replacing ECharts' model or restarting guides.
const StableTrendChart = memo(Chart);
const anchorLabel = (anchor) => `${anchor.target} · ${anchor.mode} · ${anchor.logicalTestId} · ${compactDuration(anchor.durationSeconds)} · first success ${anchor.runId} · ${formatFullDate(anchor.timestamp)}`;

function compactDuration(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  if (seconds < 60) return formatDuration(seconds);
  const rounded = Math.round(seconds);
  return `${Math.floor(rounded / 60)}m${rounded % 60}s`;
}


export default function DurationHistory({ history, filters, coverage, range, onRangeChange, onOpenBenchmarks }) {
  const theme = useTheme();
  const helpId = useId();
  const chartRef = useRef(null);
  const inspectionSource = useRef('mouse');
  const [inspection, setInspection] = useState(null);
  const lastInspection = useRef(null);
  const inspectedIndex = inspection?.history === history ? inspection.index : null;
  const eligible = history.slots.flatMap((slot, index) => slot.run && Number.isFinite(history.series[0]?.data[index]) ? [index] : []);
  const hidePointer = useCallback(() => {
    const chart = chartRef.current;
    if (chart && !chart.isDisposed()) {
      chart.dispatchAction?.({ type: 'hideTip' });
      chart.dispatchAction?.({ type: 'updateAxisPointer', currTrigger: 'leave' });
    }
  }, []);
  const inspect = useCallback((index) => {
    if (lastInspection.current?.history === history && lastInspection.current.index === index) return;
    lastInspection.current = { history, index };
    setInspection({ history, index });
    const chart = chartRef.current;
    if (chart && !chart.isDisposed()) {
      if (index == null) {
        hidePointer();
      } else if (inspectionSource.current !== 'mouse' && inspectionSource.current !== 'pen') chart.dispatchAction?.({ type: 'showTip', seriesIndex: 0, dataIndex: index });
    }
  }, [history, hidePointer]);
  const series = history.series[0];
  const inspectedSlot = inspectedIndex != null ? history.slots[inspectedIndex] : null;
  const inspectedValue = inspectedSlot ? series?.data[inspectedIndex] : null;
  const anchorsFor = useCallback((run) => (history.estimates ?? []).filter((anchor) => anchor.estimatedRunId === run?.runId), [history]);
  const inspectedAnchors = anchorsFor(inspectedSlot?.run);
  const firstValue = eligible.length ? series.data[eligible[0]] : null;
  const baseline = Number.isFinite(series?.baseline) ? series.baseline : firstValue;
  const blue = colorTokens[theme.palette.mode === 'dark' ? 'dark' : 'light'].interactive;
  const scale = Math.max(...(series?.data.filter(Number.isFinite) ?? [0]), 0) > 60 ? 1 / 60 : 1;
  const rangeState = classifyDurationChange(history.durationDelta);
  const rangeTone = changeTone(rangeState);
  const period = historyRangeById.get(range)?.period ?? 'Selected period';
  const option = useMemo(() => ({
    textStyle: { fontFamily: theme.typography.fontFamily, color: theme.palette.text.secondary },
    tooltip: {
      trigger: 'axis', triggerOn: 'mousemove|click',
      transitionDuration: 0,
      confine: true, padding: [6, 8], textStyle: { fontSize: 11, lineHeight: 16 },
      extraCssText: 'max-width:240px;white-space:normal;overflow-wrap:anywhere;pointer-events:none;',
      axisPointer: { type: 'line', snap: false, animation: false, lineStyle: { color: theme.palette.text.secondary, type: 'dotted', width: 1 } },
      formatter: (parameters) => {
        const point = (Array.isArray(parameters) ? parameters : [parameters]).find((item) => Number.isFinite(item.value?.[1]));
        const run = point && history.slots[point.dataIndex]?.run;
        if (!run) return '';
        return `<strong>${escapeHtml(shortSha(run))}</strong> · ${escapeHtml(formatFullDate(commitTimestampFor(run)))}<br/>Runtime · <strong>${compactDuration(point.value[1] / scale)}</strong>${anchorsFor(run).length ? '<br/>Normalized estimate' : ''}`;
      },
    },
    grid: { left: 40, right: 18, top: 28, bottom: 54 },
    xAxis: {
      type: 'value', min: 0, max: Math.max(1, history.axisMax),
      interval: history.axisMax > 0 ? Math.max(1, history.axisMax / Math.min(6, history.axisMax)) : 1,
      name: history.mode === 'intraday' ? 'Commit time (UTC)' : 'Date (UTC)', nameLocation: 'middle', nameGap: 36,
      axisLine: { lineStyle: { color: theme.palette.divider } }, axisTick: { show: false }, splitLine: { show: false },
      axisLabel: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily, fontSize: 11, hideOverlap: true, formatter: (value) => {
        const index = Math.round(value);
        if (history.mode === 'intraday') {
          const run = history.slots[index]?.run;
          return run ? new Date(commitTimestampFor(run)).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC' }) : '';
        }
        const day = history.dayKeys[index];
        return day ? formatShortDate(`${day}T12:00:00Z`) : '';
      } }, nameTextStyle: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily, fontSize: 11 },
    },
    yAxis: {
      type: 'value', scale: true, name: scale === 1 ? 'Seconds' : 'Minutes',
      axisLabel: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily, fontSize: 11 },
      nameTextStyle: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily },
      splitLine: { lineStyle: { color: theme.palette.divider, type: 'dotted' } },
    },
    series: series ? [{
      name: 'Selected runtime', type: 'line', encode: { x: 0, y: 1 },
      data: series.data.map((value, index) => [history.slots[index].x, Number.isFinite(value) ? value * scale : null]),
      smooth: false, connectNulls: true, showSymbol: true, symbol: 'circle', symbolSize: 4,
      lineStyle: { color: blue, width: 2.5, shadowBlur: 7, shadowOffsetY: 3, shadowColor: `${blue}30` },
      itemStyle: chartPointStyle(blue, theme.palette.background.paper),
      areaStyle: { color: chartAreaGradient(blue, 0.15) },
      markLine: {
        silent: true, symbol: 'none',
        label: { formatter: `Baseline · ${compactDuration(baseline)}`, position: 'insideEndTop', color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily },
        lineStyle: { color: theme.palette.text.disabled, type: 'dotted', width: 1.2 },
        data: [
          ...(Number.isFinite(baseline) ? [{ yAxis: baseline * scale }] : []),

        ],
      },

    }] : [],
  }), [history, series, theme, scale, blue, baseline, anchorsFor]);
  const events = useMemo(() => ({
    finished: (_, chart) => { chartRef.current = chart; },
    updateAxisPointer: (event) => {
      const value = event.axesInfo?.[0]?.value;
      // Tooltip release can emit no axis value. It is not an explicit clear.
      if (Number.isFinite(value)) {
        const index = nearestTrendIndex(history, value);
        if (index == null) hidePointer();
        else inspect(index);
      }
    },
    click: (event) => {
      if (event.componentType === 'series') {
        const index = nearestTrendIndex(history, history.slots[event.dataIndex]?.x);
        if (index == null) hidePointer();
        else inspect(index);
      }
    },
    globalout: () => { if (shouldHideTrendPointer(inspectionSource.current)) hidePointer(); },
  }), [history, inspect, hidePointer]);
  return (
    <SectionCard data-testid="performance-trend" title="Performance Trend" subtitle="Sum of selected benchmark runtimes · develop only" sx={{ height: '100%', minWidth: 0 }} contentSx={{ p: 2, '&:last-child': { pb: 2 }, height: '100%', display: 'flex', flexDirection: 'column' }} action={(
      <Box sx={{ maxWidth: '100%', overflowX: 'auto' }}><ToggleButtonGroup exclusive size="small" value={range} onChange={(_, next) => next && onRangeChange(next)} aria-label="History timeframe">
        {historyRanges.map(({ id: value, label }) => <ToggleButton key={value} value={value} aria-label={label} sx={{ minWidth: 38, minHeight: { xs: 44, sm: 32 }, px: 1, fontSize: 11 }}>{value}</ToggleButton>)}
      </ToggleButtonGroup></Box>
    )}>
      <Stack direction="row" sx={{ flexWrap: 'wrap', alignItems: 'baseline', gap: 1, mb: 0.5 }}>
        <Typography sx={{ fontFamily: monoFont, fontSize: 24, lineHeight: '32px' }}>{formatDuration(history.currentDuration)}</Typography>
        <Typography data-testid="performance-range-change" data-change-state={rangeState} variant="caption" title={`Range change · ${period} · ${shortSha(history.latestRun)} vs ${shortSha(history.firstRun)}`} sx={{ color: rangeTone === 'neutral' ? 'text.secondary' : `${rangeTone}.main` }}>{formatPercent(history.durationDelta)} in selected period</Typography>
      </Stack>
      <Typography variant="caption" color="text.secondary">{filters ? `${filters.targets.length} target${filters.targets.length === 1 ? '' : 's'} · ${filters.suites.length} suite${filters.suites.length === 1 ? '' : 's'} · ${filters.modes.join(' + ') || 'No modes'}` : history.summary}{coverage ? ` · ${coverage.completed ?? 0}/${coverage.total ?? 0} selected benchmark results` : ''}</Typography>
      {history.insufficientData || !eligible.length ? <Box data-testid="performance-trend-insufficient" sx={{ flex: 1, minHeight: 220, display: 'grid', placeItems: 'center', textAlign: 'center', color: 'text.secondary' }}>There isn't enough data for the selected time range</Box> : (
        <Box data-testid="performance-trend-chart" tabIndex={0} role="group" aria-label="Inspect performance trend" aria-describedby={helpId} onKeyDown={(event) => {
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Escape'].includes(event.key)) return;
          inspectionSource.current = 'keyboard';
          event.preventDefault(); inspect(trendKeyIndex(eligible, inspectedIndex, event.key));
        }} onPointerDown={(event) => {
          inspectionSource.current = event.pointerType;
          const chart = chartRef.current;
          if (!chart || chart.isDisposed()) return;
          const rect = event.currentTarget.getBoundingClientRect();
          const pixel = [event.clientX - rect.left, event.clientY - rect.top];
          if (!chart.containPixel('grid', pixel)) return;
          const value = chart.convertFromPixel('grid', pixel);
          const index = nearestTrendIndex(history, value?.[0]);
          if (index == null) hidePointer();
          else inspect(index);
        }} onPointerMove={(event) => { if (event.pointerType === 'mouse' || event.pointerType === 'pen') inspectionSource.current = event.pointerType; }} onPointerLeave={() => { if (shouldHideTrendPointer(inspectionSource.current)) hidePointer(); }} sx={{ position: 'relative', mt: 2, minHeight: 320, flex: 1, outlineOffset: 2, '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main' } }}>
          <StableTrendChart option={option} height={320} ariaLabel={`Performance trend for ${range}`} ariaDescribedBy={helpId} onEvents={events} lazyUpdate={false} />
        </Box>
      )}
      {inspectedSlot && <Typography component="span" role="status" aria-live="polite" sx={visuallyHiddenStyles}>{formatFullDate(commitTimestampFor(inspectedSlot.run))} · {shortSha(inspectedSlot.run)} · {compactDuration(inspectedValue)}</Typography>}
      {inspectedAnchors.length > 0 && <Box component="details" sx={{ mt: 1, fontSize: 11 }}>
        <Box component="summary" sx={{ cursor: 'pointer', py: 0.75 }}>Inspect first-success anchors ({inspectedAnchors.length})</Box>
        <Box tabIndex={0} role="region" aria-label="Selected trend estimate anchors" sx={{ maxHeight: 160, overflow: 'auto', p: 1, border: 1, borderColor: 'divider', overflowWrap: 'anywhere' }}>{inspectedAnchors.map((anchor) => <Box key={anchor.testId ?? `${anchor.target}:${anchor.mode}:${anchor.logicalTestId}`} sx={{ mb: 0.75 }}>{anchorLabel(anchor)}</Box>)}</Box>
      </Box>}
      <Typography id={helpId} variant="caption" color="text.secondary" sx={{ fontSize: 11, mt: 0.5 }}>Arrow keys inspect runs · Home/End first/last · Escape clears. Inspection stays available after leaving the graph.</Typography>
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>Trend values are normalized to the latest test catalog using fixed first-success anchors for added benchmarks. Explore original per-benchmark measurements: <Link component="button" type="button" onClick={onOpenBenchmarks} sx={{ font: 'inherit', verticalAlign: 'baseline' }}>Benchmarks</Link>.</Typography>
    </SectionCard>
  );
}
