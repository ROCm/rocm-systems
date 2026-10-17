import {
  Box,
  Alert,
  Button,
  Chip,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import SwapVertRoundedIcon from '@mui/icons-material/SwapVertRounded';
import RunSelector from '../compare/RunSelector';
import RunMetadataDiff from '../compare/RunMetadataDiff';
import ComparisonMetrics from '../compare/ComparisonMetrics';
import Chart from '../shared/Chart';
import SectionCard from '../shared/SectionCard';
import StatusChip from '../shared/StatusChip';
import { selectRunComparison } from '../../data/selectors';
import { compareRunExecution } from '../../data/runOrdering';
import {
  escapeHtml,
  formatDuration,
  formatPercent,
  shortSha,
} from '../../utils/formatters';
import { classifyDurationChange } from '../../utils/performance';
import CommitComparison from '../shared/CommitComparison';

const NOISE_TOLERANCE = 3;


function TestAvailability({ test, run, benchmark }) {
  const label = !run ? 'No run selected'
    : !run.configurations?.some(({ target, threadingMode }) => target === benchmark.target && threadingMode === benchmark.mode)
      ? 'Configuration not published' : 'Unavailable in catalog';
  return run && test
    ? <StatusChip status={test.status} />
    : <Chip size="small" variant="outlined" label={label} />;
}

function ExcludedTestsTable({ comparisons, comparableCount, baseline, candidate }) {
  const hasPair = Boolean(baseline && candidate);
  if (comparisons.length === 0) {
    return (
      <Typography sx={{ color: 'text.secondary', py: 3, textAlign: 'center' }}>
        {!hasPair ? 'Select two runs to compare.' : comparableCount > 0
          ? 'Every selected benchmark has completed data in both runs.'
          : 'No benchmarks in the selected scope.'}
      </Typography>
    );
  }

  return (
    <TableContainer tabIndex={0} role="region" aria-label="Excluded results scroll area" sx={{ maxWidth: '100%', overflowX: 'auto', border: 1, borderColor: 'divider', borderRadius: 2 }}>
      <Table size="small" sx={{ minWidth: 760 }}>
        <TableHead>
          <TableRow>
            <TableCell>Benchmark</TableCell>
            <TableCell>Target / mode</TableCell>
            <TableCell>Suite</TableCell>
            <TableCell>Baseline</TableCell>
            <TableCell>Candidate</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {comparisons.map((comparison) => {
            const benchmark = comparison.candidateTest ?? comparison.baselineTest;
            return (
              <TableRow key={benchmark.testId}>
                <TableCell sx={{ fontWeight: 700 }}>{benchmark.name}</TableCell>
                <TableCell>{benchmark.target} · {benchmark.mode}</TableCell>
                <TableCell>{benchmark.suite}</TableCell>
                <TableCell><TestAvailability run={baseline} benchmark={benchmark} test={comparison.baselineTest} /></TableCell>
                <TableCell><TestAvailability run={candidate} benchmark={benchmark} test={comparison.candidateTest} /></TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export default function CompareRunsView({
  data,
  filters,
  selectedBaselineId,
  selectedCandidateId,
  onBaselineChange,
  onCandidateChange,
  onSwap,
}) {
  const theme = useTheme();
  const compactChart = useMediaQuery(theme.breakpoints.down('sm'));
  const publishedRuns = data.allRuns ?? [];
  const candidate = publishedRuns.find((run) => run.runId === selectedCandidateId) ?? null;
  const baseline = publishedRuns.find((run) => run.runId === selectedBaselineId) ?? null;
  const viewModel = selectRunComparison(candidate, baseline, filters, NOISE_TOLERANCE);
  const hasRuns = Boolean(candidate && baseline);
  const runOptions = publishedRuns.slice().sort((a, b) => compareRunExecution(b, a));
  const chartItems = viewModel.comparable.filter((item) => Number.isFinite(item.delta));
  const maximumDelta = Math.max(...chartItems.map((item) => Math.abs(item.delta)), 0);
  const axisLimit = Math.max(5, Math.ceil(maximumDelta * 1.25));
  const chartHeight = Math.min(720, Math.max(320, chartItems.length * 34 + 100));
  const richLabelStyles = {
    separator: { color: theme.palette.text.disabled },
    target: {
      color: theme.palette.text.secondary,
      fontWeight: 700,
      fontFamily: theme.typography.fontFamily,
    },
    benchmark: {
      color: theme.palette.text.secondary,
      fontWeight: 520,
      fontFamily: theme.typography.fontFamily,
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
    return `${indicator} ${formatPercent(delta)}`;
  };
  const categoryFor = (item) => {
    const target = `${item.candidateTest.target} · ${item.candidateTest.mode}`.replace(/[{}|]/g, '');
    const benchmark = item.candidateTest.name.replace(/[{}|]/g, '');
    const separator = compactChart ? '\n' : '{separator| · }';
    return `{target|${target}}${separator}{benchmark|${benchmark}}`;
  };

  const option = {
    textStyle: { fontFamily: theme.typography.fontFamily, color: theme.palette.text.secondary },
    tooltip: {
      trigger: 'item',
      backgroundColor: theme.palette.background.paper,
      borderColor: theme.palette.divider,
      textStyle: { color: theme.palette.text.primary },
      formatter: ({ data: point }) => [
        `<strong>${escapeHtml(point.comparison.candidateTest.name)}</strong>`,
        `${escapeHtml(point.comparison.candidateTest.target)} · ${escapeHtml(point.comparison.candidateTest.mode)} · ${escapeHtml(point.comparison.candidateTest.suite)}`,
        `Candidate ${formatDuration(point.comparison.candidateTest.durationSeconds)}`,
        `Baseline ${formatDuration(point.comparison.baselineTest.durationSeconds)}`,
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
      axisLabel: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily, formatter: '{value}%' },
      nameTextStyle: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily },
      axisLine: { show: true, lineStyle: { color: theme.palette.divider } },
      splitLine: { lineStyle: { color: theme.palette.divider, type: 'dashed' } },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: chartItems.map(categoryFor),
      axisTick: { show: false },
      axisLine: { show: false },
      axisLabel: {
        fontSize: compactChart ? 10 : 11,
        fontFamily: theme.typography.fontFamily,
        color: theme.palette.text.secondary,
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
      data: chartItems.map((item) => {
        const performanceColor = colorFor(item.delta);
        return {
          value: item.delta,
          comparison: item,
          benchmarkId: item.candidateTest.logicalTestId,
          targetId: item.candidateTest.target,
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
            fontFamily: theme.typography.fontFamily,
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
    <Box data-testid="comparison-results" sx={{ display: 'grid', gap: 1.75, minWidth: 0 }}>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'minmax(0, 1fr) auto minmax(0, 1fr)' }, alignItems: 'center', gap: 1.25 }}>
        <RunSelector label="Baseline run" options={runOptions} value={baseline} onChange={onBaselineChange} />
        <Button
          size="small"
          variant="outlined"
          startIcon={<SwapVertRoundedIcon />}
          disabled={!baseline || !candidate}
          onClick={() => {
            if (onSwap) onSwap({ baselineId: candidate.runId, candidateId: baseline.runId });
            else { onBaselineChange(candidate.runId); onCandidateChange(baseline.runId); }
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
        <RunSelector label="Candidate run" options={runOptions} value={candidate} onChange={onCandidateChange} />
      </Box>
      {!candidate && selectedCandidateId && <Alert severity="info">Selected candidate is unavailable · {selectedCandidateId}. Choose a published attempt; no different run has been substituted.</Alert>}
      {!baseline && selectedBaselineId && <Alert severity="info">Selected baseline is unavailable · {selectedBaselineId}. Choose a published attempt; no different run has been substituted.</Alert>}

      <SectionCard
        title="Performance Change by Benchmark"
        subtitle={(
          <>
            <Box component="span" sx={{ display: 'block' }}>Candidate vs baseline · Lower duration is faster</Box>
            <CommitComparison candidate={candidate} baseline={baseline} sx={{ mt: 0.25 }} />
          </>
        )}
      >
        <ComparisonMetrics model={viewModel} candidate={candidate} baseline={baseline} tolerance={NOISE_TOLERANCE} />
        <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 0.75, mt: 1.25 }}>
          <Chip size="small" color="success" variant="outlined" label={`${hasRuns ? viewModel.counts.faster : '—'} faster`} />
          <Chip size="small" variant="outlined" label={`${hasRuns ? viewModel.counts.neutral : '—'} within ±${NOISE_TOLERANCE}%`} />
          <Chip size="small" color="error" variant="outlined" label={`${hasRuns ? viewModel.counts.slower : '—'} slower`} />
          {viewModel.counts.unavailable > 0 && <Chip size="small" variant="outlined" label={`${viewModel.counts.unavailable} percentage unavailable`} />}
        </Stack>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 1 }}>
          Totals use matched benchmarks with valid completed durations in both runs; bars and faster/slower/noise counts include only available percentage changes.
          Matching uses exact target + mode + workload identity, not environment equivalence. Exclusions are not zero-duration results. A zero baseline has no percentage ratio.
        </Typography>
        {chartItems.length > 0 ? (
          <Box sx={{ mx: { xs: -1.25, sm: -0.5 }, mt: 0.75 }}>
            <Chart option={option} height={chartHeight} ariaLabel="Performance change by benchmark comparison chart" />
          </Box>
        ) : (
          <Typography sx={{ color: 'text.secondary', py: 8, textAlign: 'center' }}>{!hasRuns ? 'Select baseline and candidate runs to compare benchmark results.' : viewModel.comparable.length ? 'No percentage changes available. Measured zero baselines remain in the results table below.' : 'No completed benchmark results are comparable between these runs.'}</Typography>
        )}
        {viewModel.comparable.length > 0 && (
          <Box component="details" sx={{ mt: 1 }}>
            <Box component="summary" sx={{ cursor: 'pointer', py: 1, fontSize: 12, fontWeight: 700 }}>View comparable results ({viewModel.comparable.length})</Box>
            <TableContainer tabIndex={0} role="region" aria-label="Comparable results scroll area">
              <Table size="small" aria-label="Comparable benchmark results" sx={{ minWidth: 540 }}>
                <TableHead><TableRow>
                  <TableCell>Benchmark / target / mode</TableCell><TableCell>Baseline</TableCell><TableCell>Candidate</TableCell><TableCell>Change</TableCell>
                </TableRow></TableHead>
                <TableBody>{viewModel.comparable.map((item) => (
                  <TableRow key={item.candidateTest.testId}>
                    <TableCell sx={{ overflowWrap: 'anywhere', maxWidth: 320 }}>{item.candidateTest.name} · {item.candidateTest.target} · {item.candidateTest.mode}</TableCell>
                    <TableCell>{formatDuration(item.baselineTest.durationSeconds)}</TableCell>
                    <TableCell>{formatDuration(item.candidateTest.durationSeconds)}</TableCell>
                    <TableCell sx={{ color: colorFor(item.delta), whiteSpace: 'nowrap' }}>{formatPercent(item.delta)} · {Number.isFinite(item.delta) ? stateFor(item.delta) === 'neutral' ? `within ±${NOISE_TOLERANCE}%` : stateFor(item.delta) : 'percentage unavailable'}</TableCell>
                  </TableRow>
                ))}</TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </SectionCard>

      <SectionCard
        title="Compared Run Information"
        subtitle="Completeness and catalog availability reflect the active target, suite and mode filters"
      >
        <RunMetadataDiff baseline={baseline} candidate={candidate} filters={filters} />
      </SectionCard>

      <SectionCard
        title="Tests Not Included in Comparison"
        subtitle="Catalog availability and benchmark-result status for the active target, suite and mode filters"
      >
        <ExcludedTestsTable comparisons={viewModel.notComparable} comparableCount={viewModel.comparable.length} baseline={baseline} candidate={candidate} />
      </SectionCard>
    </Box>
  );
}
