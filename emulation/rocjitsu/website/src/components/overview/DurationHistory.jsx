import {
  Alert,
  Box,
  Chip,
  Link,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  useTheme,
} from '@mui/material';
import ArrowDownwardRoundedIcon from '@mui/icons-material/ArrowDownwardRounded';
import ArrowUpwardRoundedIcon from '@mui/icons-material/ArrowUpwardRounded';
import RemoveRoundedIcon from '@mui/icons-material/RemoveRounded';
import Chart from '../shared/Chart';
import SectionCard from '../shared/SectionCard';
import { escapeHtml, formatDuration, formatPercent, formatShortDate, shortSha } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';
import { chartAreaGradient, chartLineStyle, chartPointStyle } from '../../utils/chartStyles';
import CommitComparison from '../shared/CommitComparison';
import { commitTimestampFor } from '../../data/runOrdering';

const ranges = [
  { value: '1D', label: '1D', ariaLabel: 'Trailing 24 hours' },
  { value: '1W', label: '1W', ariaLabel: 'Trailing 7 days' },
  { value: '1M', label: '1M', ariaLabel: 'Trailing 30 days' },
  { value: '3M', label: '3M', ariaLabel: 'Trailing 90 days' },
  { value: '6M', label: '6M', ariaLabel: 'Trailing 180 days' },
  { value: 'YTD', label: 'YTD', ariaLabel: 'Year to date' },
  { value: 'ALL', label: 'ALL', ariaLabel: 'All available history' },
];

const MINUTES_THRESHOLD_SECONDS = 60;

function axisDecimalPlaces(range) {
  if (!(range > 0)) return 0;
  return Math.max(0, Math.ceil(-Math.log10(range / 5)));
}

function formatCompactDuration(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  if (seconds < MINUTES_THRESHOLD_SECONDS) return formatDuration(seconds);
  const roundedSeconds = Math.round(seconds);
  return `${Math.floor(roundedSeconds / 60)}m${roundedSeconds % 60}s`;
}

function rangePeriodLabel(range) {
  return {
    '1D': 'past day',
    '1W': 'past week',
    '1M': 'past month',
    '3M': 'past 3 months',
    '6M': 'past 6 months',
    YTD: 'this year',
    ALL: 'all available history',
  }[range] ?? 'Selected period';
}

export default function DurationHistory({
  history,
  range,
  onRangeChange,
  onOpenBenchmarks,
  showNormalizationNote,
}) {
  const theme = useTheme();
  const axisColor = theme.palette.text.secondary;
  const gridColor = theme.palette.divider;
  const hasDelta = Number.isFinite(history.durationDelta);
  const rangeState = classifyDurationChange(history.durationDelta);
  const rangeTone = changeTone(rangeState);
  const rangeColor = rangeTone === 'neutral' ? 'text.secondary' : `${rangeTone}.main`;
  const useContinuousDateAxis = true;
  const labelCount = Math.min(7, Math.max(history.axisMax, 1));
  const maximumDuration = Math.max(...history.series.flatMap((series) => (
    series.data.filter(Number.isFinite)
  )), 0);
  const showMinutes = maximumDuration > MINUTES_THRESHOLD_SECONDS;
  const durationScale = showMinutes ? 1 / 60 : 1;
  const finiteDataValues = history.series.flatMap((series) => (
    series.data.filter(Number.isFinite)
  ));
  const finiteBaselineValues = history.series
    .map((series) => series.baseline)
    .filter(Number.isFinite);
  const scaledDataValues = finiteDataValues.map((value) => value * durationScale);
  const scaledBaselineValues = finiteBaselineValues.map((value) => value * durationScale);
  const yAxisMinValue = Math.min(...scaledDataValues, ...scaledBaselineValues);
  const yAxisMaxValue = Math.max(...scaledDataValues, ...scaledBaselineValues);
  const yAxisDecimalPlaces = axisDecimalPlaces(yAxisMaxValue - yAxisMinValue);
  const yAxisPrecision = 10 ** yAxisDecimalPlaces;
  const yAxisMin = Number.isFinite(yAxisMinValue)
    ? Math.floor(yAxisMinValue * yAxisPrecision) / yAxisPrecision
    : undefined;
  const yAxisMax = Number.isFinite(yAxisMaxValue)
    ? Math.ceil(yAxisMaxValue * yAxisPrecision) / yAxisPrecision
    : undefined;
  const option = {
    color: history.series.map((series) => series.color),
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'line', lineStyle: { color: theme.palette.text.disabled, type: 'dashed' } },
      backgroundColor: theme.palette.background.paper,
      borderColor: theme.palette.divider,
      textStyle: { color: theme.palette.text.primary },
      formatter: (points) => {
        const usable = points
          .map((point) => ({
            ...point,
            duration: Array.isArray(point.value) ? point.value[1] : point.value,
          }))
          .filter((point) => Number.isFinite(point.duration));
        if (!usable.length) return '';
        const run = history.slots[usable[0].dataIndex]?.run;
        return [
          `<strong>Commit date · ${escapeHtml(formatShortDate(commitTimestampFor(run)))}</strong>`,
          `Commit SHA · ${escapeHtml(shortSha(run))}`,
          `Test catalog · ${escapeHtml(run.catalogId ?? 'unknown')}`,
          ...usable.map((point) => {
            const series = history.series.find((candidate) => candidate.target === point.seriesName);
            const duration = point.duration / durationScale;
            const perfChange = Number.isFinite(series?.baseline) && series.baseline !== 0
              ? ((duration - series.baseline) / series.baseline) * 100
              : null;
            return `${point.marker}${escapeHtml(point.seriesName)}&nbsp;&nbsp;<strong>${formatCompactDuration(duration)}</strong>`
              + `${Number.isFinite(perfChange) ? ` · Time change ${formatPercent(perfChange)}` : ''}`
              + `${Number.isFinite(series?.baseline) ? ` · Base time ${formatCompactDuration(series.baseline)}` : ''}`;
          }),
        ].join('<br/>');
      },
    },
    grid: { left: 54, right: 18, top: 18, bottom: 58 },
    xAxis: {
      type: useContinuousDateAxis ? 'value' : 'category',
      name: history.mode === 'intraday' ? 'Commit time (UTC)' : 'Date (UTC)',
      nameLocation: 'middle',
      nameGap: 43,
      boundaryGap: false,
      ...(useContinuousDateAxis ? {
        min: 0,
        max: history.axisMax,
        // A zero-width axis holds a single slot, and without an explicit interval
        // ECharts spreads its default tick count over it and repeats that slot's label.
        interval: history.axisMax > 0 ? history.axisMax / labelCount : 1,
      } : {
        data: history.slots.map((slot) => slot.label),
      }),
      axisLine: { lineStyle: { color: gridColor } },
      axisTick: { show: false },
      axisLabel: {
        color: axisColor,
        fontSize: 10,
        lineHeight: 14,
        hideOverlap: true,
        ...(useContinuousDateAxis ? {
          formatter: (value) => {
            if (history.mode === 'intraday') {
              const run = history.slots[Math.round(value)]?.run;
              return run ? new Date(commitTimestampFor(run)).toLocaleTimeString(undefined, {
                hour: '2-digit',
                minute: '2-digit',
                hour12: false,
                timeZone: 'UTC',
              }) : '';
            }
            const dayKey = history.dayKeys[Math.round(value)];
            return dayKey ? formatShortDate(`${dayKey}T12:00:00Z`) : '';
          },
        } : {}),
      },
      nameTextStyle: { color: axisColor, fontSize: 11 },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: showMinutes ? 'Minutes' : 'Seconds',
      scale: true,
      ...(Number.isFinite(yAxisMin) ? { min: yAxisMin } : {}),
      ...(Number.isFinite(yAxisMax) ? { max: yAxisMax } : {}),
      nameTextStyle: { color: axisColor, align: 'right', padding: [0, 2, 6, 0] },
      axisLine: { show: false },
      axisLabel: { color: axisColor, fontSize: 11 },
      splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
    },
    series: history.series.map((series) => {
      const firstIndex = series.data.findIndex((value) => Number.isFinite(value));
      const firstValue = firstIndex >= 0 ? series.data[firstIndex] : null;
      const baselineValue = Number.isFinite(series.baseline) ? series.baseline : firstValue;
      const latestIndex = series.data.reduce((lastIndex, value, index) => (
        Number.isFinite(value) ? index : lastIndex
      ), -1);
      const latestValue = latestIndex >= 0 ? series.data[latestIndex] : null;
      return {
        name: series.target,
        type: 'line',
        ...(useContinuousDateAxis ? { encode: { x: 0, y: 1 } } : {}),
        data: useContinuousDateAxis
          ? series.data.map((value, index) => [
            history.slots[index].x,
            Number.isFinite(value) ? value * durationScale : null,
          ])
          : series.data.map((value) => (
            Number.isFinite(value) ? value * durationScale : value
          )),
        smooth: 0.12,
        showSymbol: false,
        symbol: 'circle',
        symbolSize: 6,
        connectNulls: true,
        lineStyle: chartLineStyle(series.color, 2.8),
        itemStyle: chartPointStyle(series.color, theme.palette.background.paper),
        emphasis: { focus: 'series', scale: 1.55, lineStyle: { width: 3.4 } },
        markLine: Number.isFinite(baselineValue) ? {
          silent: true,
          symbol: 'none',
          label: {
            show: true,
            formatter: `— (${formatCompactDuration(baselineValue)}) —`,
            position: 'insideEndTop',
          },
          lineStyle: { color: theme.palette.text.disabled, type: 'dashed', width: 1.25 },
          data: [{ yAxis: baselineValue * durationScale }],
        } : undefined,
        areaStyle: {
          color: chartAreaGradient(series.color, history.series.length === 1 ? 0.24 : 0.11),
          opacity: 1,
        },
        markPoint: Number.isFinite(latestValue) ? {
          silent: true,
          symbol: 'circle',
          symbolSize: 12,
          label: { show: false },
          itemStyle: { ...chartPointStyle(series.color, theme.palette.background.paper), borderWidth: 3 },
          data: [{ coord: [history.slots[latestIndex].x, latestValue * durationScale] }],
        } : undefined,
      };
    }),
  };

  return (
    <SectionCard
      data-testid="performance-trend"
      title="Performance Trend"
      subtitle={(
        <>
          <Box component="span" sx={{ display: 'block' }}>
            This graph samples data from:
          </Box>
          <Box component="span" sx={{ display: 'block' }}>
            commits from the <u>default branch</u> where <u>all tests passed</u>.
          </Box>
        </>
      )}
      sx={{ height: '100%' }}
      contentSx={{ height: '100%', boxSizing: 'border-box', display: 'flex', flexDirection: 'column' }}
      action={(
        <Box sx={{ maxWidth: '100%', overflowX: 'auto', pb: 0.25 }}>
          <ToggleButtonGroup
            exclusive
            size="small"
            value={range}
            onChange={(_, nextRange) => nextRange && onRangeChange(nextRange)}
            aria-label="History timeframe"
          >
            {ranges.map((item) => (
              <ToggleButton
                key={item.value}
                value={item.value}
                aria-label={item.ariaLabel}
                sx={{ minWidth: 38, px: { xs: 0.8, sm: 1.1 }, py: 0.45, fontSize: 11 }}
              >
                {item.label}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>
      )}
    >
      <Stack direction={{ xs: 'column', sm: 'row' }} sx={{ justifyContent: 'space-between', alignItems: { xs: 'flex-start', sm: 'flex-end' }, gap: 1.5, mb: 0.75 }}>
        <Stack direction="row" sx={{ alignItems: 'flex-end', flexWrap: 'wrap', gap: { xs: 2, sm: 2.5 } }}>
          <Box>
            <Typography variant="overline" sx={{ color: 'text.secondary' }}>
              Range change
            </Typography>
            <Stack
              data-testid="performance-range-change"
              data-change-state={rangeState}
              direction="row"
              sx={{ alignItems: 'flex-end', color: rangeColor, gap: 0.35, mt: 0.2 }}
            >
              {rangeState === 'faster' && <ArrowDownwardRoundedIcon sx={{ fontSize: 24 }} />}
              {rangeState === 'slower' && <ArrowUpwardRoundedIcon sx={{ fontSize: 24 }} />}
              {rangeState === 'neutral' && <RemoveRoundedIcon sx={{ fontSize: 24 }} />}
              <Typography sx={{ fontSize: 25, lineHeight: 1, fontWeight: 820, letterSpacing: '-.035em' }}>
                {hasDelta ? formatPercent(Math.abs(history.durationDelta), false) : '—'}
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary', ml: 0.35, mb: 0.1, lineHeight: 1 }}>
                {rangePeriodLabel(range)}
              </Typography>
            </Stack>
            {hasDelta && (
              <CommitComparison candidate={history.latestRun} baseline={history.firstRun} sx={{ mt: 0.25 }} />
            )}
          </Box>
          <Box sx={{ borderLeft: 1, borderColor: 'divider', pl: 2 }}>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Latest selected total
            </Typography>
            <Typography sx={{ fontSize: 20, lineHeight: 1.15, fontWeight: 750, letterSpacing: '-.025em', mt: 0.35 }}>
              {formatDuration(history.currentDuration)}
            </Typography>
          </Box>
        </Stack>
        <Box sx={{ textAlign: { sm: 'right' } }}>
          <Stack direction="row" sx={{ flexWrap: 'wrap', justifyContent: { sm: 'flex-end' }, gap: 0.65 }}>
            {history.series.map((series) => (
              <Chip
                key={series.target}
                size="small"
                label={series.target}
                sx={{ '&::before': { content: '""', width: 7, height: 7, borderRadius: '50%', bgcolor: series.color, ml: 1 } }}
              />
            ))}
          </Stack>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {history.summary}
          </Typography>
        </Box>
      </Stack>
      {history.insufficientData ? (
        <Box
          data-testid="performance-trend-insufficient"
          sx={{ flex: 1, minHeight: 278, display: 'grid', placeItems: 'center', color: 'text.secondary', textAlign: 'center' }}
        >
          There isn't enough data for the selected time range
        </Box>
      ) : (
        <Box data-testid="performance-trend-chart" sx={{ mx: -0.75, flex: 1, minHeight: 278 }}>
          <Chart option={option} height="100%" ariaLabel={`Performance trend for ${range}`} />
        </Box>
      )}
      {showNormalizationNote && (
        <Alert
          data-testid="performance-trend-normalization-note"
          severity="info"
          variant="outlined"
          sx={{
            mt: 1,
            py: 0,
            alignItems: 'center',
            '& .MuiAlert-message': {
              py: 0.75,
              display: 'flex',
              alignItems: 'center',
              lineHeight: 1.5,
            },
          }}
        >
          <Box component="span">
            Performance trend values are normalized to the latest test catalog. Original total
            duration for each run is available in the{' '}
            <Link
              component="button"
              type="button"
              onClick={onOpenBenchmarks}
              underline="hover"
              sx={{
                p: 0,
                border: 0,
                font: 'inherit',
                verticalAlign: 'baseline',
                cursor: 'pointer',
              }}
            >
              Benchmarks
            </Link>
            {' '}tab.
          </Box>
        </Alert>
      )}
    </SectionCard>
  );
}
