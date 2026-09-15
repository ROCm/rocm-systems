import {
  Box,
  Button,
  Chip,
  Stack,
  Typography,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import ArrowDownwardRoundedIcon from '@mui/icons-material/ArrowDownwardRounded';
import ArrowUpwardRoundedIcon from '@mui/icons-material/ArrowUpwardRounded';
import RemoveRoundedIcon from '@mui/icons-material/RemoveRounded';
import SwapVertRoundedIcon from '@mui/icons-material/SwapVertRounded';
import RunSelector from '../compare/RunSelector';
import Chart from '../shared/Chart';
import SectionCard from '../shared/SectionCard';
import { previousCompletedRun, selectRunComparison } from '../../data/selectors';
import { escapeHtml, formatDuration, formatPercent, shortSha } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';
import CommitComparison from '../shared/CommitComparison';

const NOISE_TOLERANCE = 3;

function SummaryStat({ label, value, color = 'text.primary' }) {
  return (
    <Box sx={{ p: 1.35, border: 1, borderColor: 'divider', borderRadius: 2, bgcolor: 'action.hover' }}>
      <Typography variant="caption" color="text.secondary">{label}</Typography>
      <Typography sx={{ fontSize: 20, fontWeight: 770, color, lineHeight: 1.2, mt: 0.3 }}>{value}</Typography>
    </Box>
  );
}

function AggregateChange({ value, candidate, baseline }) {
  const state = classifyDurationChange(value, NOISE_TOLERANCE);
  const tone = changeTone(state);
  const color = tone === 'neutral' ? 'text.secondary' : `${tone}.main`;
  return (
    <Box sx={{ p: 1.35, border: 1, borderColor: 'divider', borderRadius: 2, bgcolor: 'action.hover' }}>
      <Typography variant="caption" color="text.secondary">Aggregate change</Typography>
      <Stack direction="row" sx={{ alignItems: 'center', color, mt: 0.3 }}>
        {state === 'faster' && <ArrowDownwardRoundedIcon sx={{ fontSize: 22 }} />}
        {state === 'slower' && <ArrowUpwardRoundedIcon sx={{ fontSize: 22 }} />}
        {state === 'neutral' && <RemoveRoundedIcon sx={{ fontSize: 22 }} />}
        <Typography sx={{ fontSize: 24, fontWeight: 820, lineHeight: 1.1 }}>
          {Number.isFinite(value) ? formatPercent(Math.abs(value), false) : '—'}
        </Typography>
      </Stack>
      {Number.isFinite(value) && <CommitComparison candidate={candidate} baseline={baseline} sx={{ mt: 0.3, fontSize: 10 }} />}
    </Box>
  );
}

export default function CompareRunsView({
  data,
  filters,
  selectedBaselineId,
  selectedCandidateId,
  onBaselineChange,
  onCandidateChange,
}) {
  const theme = useTheme();
  const compactChart = useMediaQuery(theme.breakpoints.down('sm'));
  const candidate = data.runs.find((run) => run.runId === selectedCandidateId) ?? data.latestCompletedRun;
  const defaultBaseline = previousCompletedRun(data.runs, candidate);
  const baseline = data.runs.find((run) => run.runId === selectedBaselineId) ?? defaultBaseline;
  const viewModel = selectRunComparison(candidate, baseline, filters, NOISE_TOLERANCE);
  const runOptions = data.runs.slice().reverse();
  const maximumDelta = Math.max(...viewModel.comparable.map((item) => Math.abs(item.delta)), 0);
  const axisLimit = Math.max(5, Math.ceil(maximumDelta * 1.25));
  const chartHeight = Math.min(720, Math.max(320, viewModel.comparable.length * 34 + 100));
  const richLabelStyles = {
    separator: { color: theme.palette.text.disabled },
    target: {
      color: theme.palette.mode === 'dark' ? '#A9BCD0' : '#3F586E',
      fontWeight: 700,
    },
    benchmark: {
      color: theme.palette.text.secondary,
      fontWeight: 520,
    },
  };

  const stateFor = (delta) => classifyDurationChange(delta, NOISE_TOLERANCE);
  const colorFor = (delta) => {
    const state = stateFor(delta);
    if (state === 'slower') return theme.palette.error.main;
    if (state === 'faster') return theme.palette.success.main;
    return theme.palette.text.disabled;
  };
  const labelFor = (delta) => {
    const state = stateFor(delta);
    const indicator = state === 'slower' ? '↑' : state === 'faster' ? '↓' : '—';
    return `${indicator} ${formatPercent(Math.abs(delta), false)}`;
  };
  const categoryFor = (item) => {
    const target = item.test.target.replace(/[{}|]/g, '');
    const benchmark = item.test.name.replace(/[{}|]/g, '');
    const separator = compactChart ? '\n' : '{separator| · }';
    return `{target|${target}}${separator}{benchmark|${benchmark}}`;
  };

  const option = {
    tooltip: {
      trigger: 'item',
      backgroundColor: theme.palette.background.paper,
      borderColor: theme.palette.divider,
      textStyle: { color: theme.palette.text.primary },
      formatter: ({ data: point }) => [
        `<strong>${escapeHtml(point.comparison.test.name)}</strong>`,
        `${escapeHtml(point.comparison.test.target)} · ${escapeHtml(point.comparison.test.suite)}`,
        `Candidate ${formatDuration(point.comparison.test.durationSeconds)}`,
        `Baseline ${formatDuration(point.comparison.previous.durationSeconds)}`,
        `Change ${formatPercent(point.comparison.delta)}`,
        `Commits ${escapeHtml(shortSha(candidate))} vs ${escapeHtml(shortSha(baseline))}`,
      ].join('<br/>'),
    },
    grid: compactChart
      ? { left: 118, right: 12, top: 18, bottom: 48 }
      : { left: 235, right: 88, top: 18, bottom: 48 },
    xAxis: {
      type: 'value',
      name: 'Duration change (%)',
      nameLocation: 'middle',
      nameGap: 32,
      min: -axisLimit,
      max: axisLimit,
      axisLabel: { color: theme.palette.text.secondary, formatter: '{value}%' },
      axisLine: { show: true, lineStyle: { color: theme.palette.divider } },
      splitLine: { lineStyle: { color: theme.palette.divider, type: 'dashed' } },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: viewModel.comparable.map(categoryFor),
      axisTick: { show: false },
      axisLine: { show: false },
      axisLabel: {
        fontSize: compactChart ? 10 : 11,
        width: compactChart ? 100 : 205,
        overflow: 'truncate',
        formatter: (value) => value,
        rich: richLabelStyles,
      },
    },
    series: [{
      type: 'bar',
      barMaxWidth: 18,
      showBackground: true,
      backgroundStyle: { color: theme.palette.action.hover, borderRadius: 6 },
      data: viewModel.comparable.map((item) => {
        const performanceColor = colorFor(item.delta);
        return {
          value: item.delta,
          comparison: item,
          benchmarkId: item.test.logicalTestId,
          targetId: item.test.target,
          itemStyle: {
            color: performanceColor,
            borderRadius: item.delta >= 0 ? [0, 5, 5, 0] : [5, 0, 0, 5],
            shadowBlur: 5,
            shadowColor: alpha(performanceColor, 0.24),
          },
          label: {
            show: true,
            position: compactChart ? 'inside' : item.delta >= 0 ? 'right' : 'left',
            color: compactChart
              ? theme.palette.getContrastText(performanceColor)
              : performanceColor,
            fontSize: compactChart ? 10 : 12,
            fontWeight: 700,
            formatter: labelFor(item.delta),
          },
        };
      }),
      emphasis: {
        itemStyle: {
          shadowBlur: 12,
          shadowColor: alpha(theme.palette.text.primary, 0.2),
        },
      },
      animationDelay: (index) => index * 18,
      markLine: {
        silent: true,
        symbol: 'none',
        label: { show: false },
        lineStyle: { color: theme.palette.text.secondary, width: 1.25 },
        data: [{ xAxis: 0 }],
      },
    }],
  };

  return (
    <Box sx={{ display: 'grid', gap: 1.75 }}>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'minmax(0, 1fr) auto minmax(0, 1fr)' }, alignItems: 'center', gap: 1.25 }}>
        <RunSelector label="Candidate run" options={runOptions} value={candidate} onChange={onCandidateChange} />
        <Button
          size="small"
          variant="outlined"
          startIcon={<SwapVertRoundedIcon />}
          disabled={!baseline || !candidate}
          onClick={() => {
            onBaselineChange(candidate.runId);
            onCandidateChange(baseline.runId);
          }}
          sx={{
            whiteSpace: 'nowrap',
            justifySelf: 'center',
            alignSelf: { xs: 'center', md: 'start' },
            mt: { xs: 0, md: 0.625 },
          }}
        >
          Swap
        </Button>
        <RunSelector label="Baseline run" options={runOptions} value={baseline} onChange={onBaselineChange} />
      </Box>

      <SectionCard
        title="Performance Change by Benchmark"
        subtitle={(
          <>
            <Box component="span" sx={{ display: 'block' }}>Candidate vs baseline · Lower duration is faster</Box>
            <CommitComparison candidate={candidate} baseline={baseline} sx={{ mt: 0.25 }} />
          </>
        )}
      >
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', md: '1.15fr repeat(4, minmax(0, 1fr))' }, gap: 1 }}>
          <AggregateChange value={viewModel.aggregateDelta} candidate={candidate} baseline={baseline} />
          <SummaryStat label="Candidate total" value={formatDuration(viewModel.candidateDuration)} />
          <SummaryStat label="Baseline total" value={formatDuration(viewModel.baselineDuration)} />
          <SummaryStat label="Comparable" value={viewModel.comparable.length} />
          <SummaryStat label="Not comparable" value={viewModel.notComparable.length} />
        </Box>
        <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 0.75, mt: 1.25 }}>
          <Chip size="small" color="success" variant="outlined" label={`${viewModel.counts.faster} faster`} />
          <Chip size="small" variant="outlined" label={`${viewModel.counts.neutral} within ±${NOISE_TOLERANCE}%`} />
          <Chip size="small" color="error" variant="outlined" label={`${viewModel.counts.slower} slower`} />
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          Only benchmarks with valid completed durations in both runs are compared.
        </Typography>
        {viewModel.notComparable.length > 0 && (
          <Box sx={{ mt: 1, p: 1.25, borderRadius: 2, border: 1, borderColor: 'divider', bgcolor: 'action.hover' }}>
            <Typography variant="caption" fontWeight={750} color="text.primary">Not included in this comparison</Typography>
            {viewModel.notComparable.map((item) => (
              <Typography key={item.test.testId} variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.35 }}>
                {item.test.target} · {item.test.name} — {item.notComparableReason}
              </Typography>
            ))}
          </Box>
        )}
        {viewModel.comparable.length > 0 ? (
          <Box sx={{ mx: { xs: -1.25, sm: -0.5 }, mt: 0.75 }}>
            <Chart option={option} height={chartHeight} ariaLabel="Performance change by benchmark comparison chart" />
          </Box>
        ) : (
          <Typography color="text.secondary" sx={{ py: 8, textAlign: 'center' }}>No completed benchmark results are comparable between these runs.</Typography>
        )}
      </SectionCard>
    </Box>
  );
}
