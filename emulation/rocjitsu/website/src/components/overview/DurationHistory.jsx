import {
  Box,
  Chip,
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
import { escapeHtml, formatDuration, formatFullDate, formatPercent } from '../../utils/formatters';
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

function visibleLabelIndexes(count, maximum = 8) {
  if (count <= maximum) return new Set(Array.from({ length: count }, (_, index) => index));
  return new Set(Array.from({ length: maximum }, (_, index) => (
    Math.round(index * (count - 1) / (maximum - 1))
  )));
}

export default function DurationHistory({ history, range, onRangeChange }) {
  const theme = useTheme();
  const axisColor = theme.palette.text.secondary;
  const gridColor = theme.palette.divider;
  const hasDelta = Number.isFinite(history.durationDelta);
  const rangeState = classifyDurationChange(history.durationDelta);
  const rangeTone = changeTone(rangeState);
  const rangeColor = rangeTone === 'neutral' ? 'text.secondary' : `${rangeTone}.main`;
  const labelIndexes = visibleLabelIndexes(history.slots.length);
  const option = {
    color: history.series.map((series) => series.color),
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'line', lineStyle: { color: theme.palette.text.disabled, type: 'dashed' } },
      backgroundColor: theme.palette.background.paper,
      borderColor: theme.palette.divider,
      textStyle: { color: theme.palette.text.primary },
      formatter: (points) => {
        const usable = points.filter((point) => Number.isFinite(point.value));
        if (!usable.length) return '';
        const run = history.slots[usable[0].dataIndex]?.run;
        const timeDetails = history.mode === 'intraday'
          ? [
            `<strong>Run time · ${escapeHtml(formatFullDate(run.timestamp))}</strong>`,
            `Commit time · ${escapeHtml(formatFullDate(commitTimestampFor(run)))}`,
          ]
          : [
            `<strong>Commit time · ${escapeHtml(formatFullDate(commitTimestampFor(run)))}</strong>`,
            `Run time · ${escapeHtml(formatFullDate(run.timestamp))}`,
          ];
        return [
          ...timeDetails,
          `Commit ${escapeHtml(run.provenance?.rocjitsuCommitSha?.slice(0, 8) ?? 'unknown')}`,
          ...usable.map((point) => `${point.marker}${escapeHtml(point.seriesName)}&nbsp;&nbsp;<strong>${formatDuration(point.value)}</strong>`),
        ].join('<br/>');
      },
    },
    grid: { left: 54, right: 18, top: 18, bottom: 58 },
    xAxis: {
      type: 'category',
      name: history.mode === 'intraday' ? 'Execution Time (UTC)' : 'Commit Date (UTC)',
      nameLocation: 'middle',
      nameGap: 43,
      boundaryGap: false,
      data: history.slots.map((slot) => slot.label),
      axisLine: { lineStyle: { color: gridColor } },
      axisTick: { show: false },
      axisLabel: {
        color: axisColor,
        fontSize: 10,
        lineHeight: 14,
        hideOverlap: true,
        interval: (index) => labelIndexes.has(index),
      },
      nameTextStyle: { color: axisColor, fontSize: 11 },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: 'Seconds',
      scale: true,
      nameTextStyle: { color: axisColor, align: 'right', padding: [0, 2, 6, 0] },
      axisLabel: { color: axisColor, fontSize: 11 },
      splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
    },
    series: history.series.map((series) => {
      const latestIndex = series.data.reduce((lastIndex, value, index) => (
        Number.isFinite(value) ? index : lastIndex
      ), -1);
      const latestValue = latestIndex >= 0 ? series.data[latestIndex] : null;
      return {
        name: series.target,
        type: 'line',
        data: series.data,
        smooth: 0.12,
        showSymbol: false,
        symbol: 'circle',
        symbolSize: 6,
        connectNulls: true,
        lineStyle: chartLineStyle(series.color, 2.8),
        itemStyle: chartPointStyle(series.color, theme.palette.background.paper),
        emphasis: { focus: 'series', scale: 1.55, lineStyle: { width: 3.4 } },
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
          data: [{ coord: [latestIndex, latestValue] }],
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
          <Box component="span" sx={{ display: 'block' }}>Selected-suite duration by target. Historical reruns are excluded.</Box>
          <Box component="span" sx={{ display: 'block' }}>{history.description}.</Box>
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
            <Typography variant="overline" color="text.secondary">Range change</Typography>
            <Stack
              data-testid="performance-range-change"
              data-change-state={rangeState}
              direction="row"
              sx={{ alignItems: 'center', color: rangeColor, gap: 0.35, mt: 0.2 }}
            >
              {rangeState === 'faster' && <ArrowDownwardRoundedIcon sx={{ fontSize: 24 }} />}
              {rangeState === 'slower' && <ArrowUpwardRoundedIcon sx={{ fontSize: 24 }} />}
              {rangeState === 'neutral' && <RemoveRoundedIcon sx={{ fontSize: 24 }} />}
              <Typography sx={{ fontSize: 25, lineHeight: 1, fontWeight: 820, letterSpacing: '-.035em' }}>
                {hasDelta ? formatPercent(Math.abs(history.durationDelta), false) : '—'}
              </Typography>
            </Stack>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.55 }}>
              {history.comparisonLabel}
            </Typography>
            {hasDelta && (
              <CommitComparison candidate={history.latestRun} baseline={history.firstRun} sx={{ mt: 0.25 }} />
            )}
          </Box>
          <Box sx={{ borderLeft: 1, borderColor: 'divider', pl: 2 }}>
            <Typography variant="caption" color="text.secondary">Latest selected total</Typography>
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
          <Typography variant="caption" color="text.secondary">
            {history.summary}
          </Typography>
        </Box>
      </Stack>
      <Box data-testid="performance-trend-chart" sx={{ mx: -0.75, flex: 1, minHeight: 278 }}>
        <Chart option={option} height="100%" ariaLabel={`Performance trend for ${range}`} />
      </Box>
    </SectionCard>
  );
}
