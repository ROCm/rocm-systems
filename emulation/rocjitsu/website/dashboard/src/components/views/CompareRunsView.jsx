import { tableStyles, numericTableCell, changedRowSx } from '../../theme/tableStyles';
import { fontSizes, fontWeights } from '../../theme/typography';
import { Box, Alert, Button, Chip, Stack, Table, TableBody, TableCell, TableHead, TableRow, Typography, useMediaQuery, useTheme } from '@mui/material';
import { alpha } from '@mui/material/styles';
import SwapVertRoundedIcon from '@mui/icons-material/SwapVertRounded';
import RunSelector from '../compare/RunSelector';
import RunMetadataDiff from '../compare/RunMetadataDiff';
import ComparisonMetrics from '../compare/ComparisonMetrics';
import Chart from '../shared/Chart';
import SectionCard from '../shared/SectionCard';
import StatusChip from '../shared/StatusChip';
import ChangeCountChip from '../shared/ChangeCountChip';
import { selectRunComparison } from '../../data/selectors';
import { compareRunExecution } from '../../data/runOrdering';
import { hasConfiguration } from '../../data/runSchema';
import {
  escapeHtml,
  formatDuration,
  formatPercent,
  shortSha,
} from '../../utils/formatters';
import { classifyDurationChange } from '../../utils/performance';
import CommitComparison from '../shared/CommitComparison';
import { colorTokens } from '../../theme/tokens';
import { targetTextColor } from '../../theme/targetColors';
import { policySwatchImage, SWATCH_HEIGHT, SWATCH_WIDTH } from '../../utils/chartStyles';
import Disclosure from '../shared/Disclosure';
import ScrollTableContainer from '../shared/ScrollTableContainer';

function TestAvailability({ test, run, benchmark }) {
  const label = !run ? 'No run selected'
    : !hasConfiguration(run, benchmark.target, benchmark.mode)
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
    <ScrollTableContainer tabIndex={0} role="region" aria-label="Excluded results scroll area" sx={{ maxHeight: 250 }}>
      <Table size="small" stickyHeader aria-label="Excluded benchmark results" sx={{ ...tableStyles.normal, minWidth: 760 }}>
        <TableHead>
          <TableRow>
            <TableCell>Benchmark</TableCell>
            <TableCell>Target / thread policy</TableCell>
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
                <TableCell sx={{ fontWeight: fontWeights.emphasized }}>{benchmark.name}</TableCell>
                <TableCell>{benchmark.target} · {benchmark.mode}</TableCell>
                <TableCell>{benchmark.suite}</TableCell>
                <TableCell><TestAvailability run={baseline} benchmark={benchmark} test={comparison.baselineTest} /></TableCell>
                <TableCell><TestAvailability run={candidate} benchmark={benchmark} test={comparison.candidateTest} /></TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </ScrollTableContainer>
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
  const viewModel = selectRunComparison(candidate, baseline, filters);
  const hasRuns = Boolean(candidate && baseline);
  const runOptions = publishedRuns.slice().sort((a, b) => compareRunExecution(b, a));
  const chartItems = viewModel.comparable.filter((item) => Number.isFinite(item.delta));
  const maximumDelta = Math.max(...chartItems.map((item) => Math.abs(item.delta)), 0);
  const axisLimit = Math.max(5, Math.ceil(maximumDelta * 1.25));
  const chartRangeAvailable = Number.isFinite(axisLimit) && axisLimit > 0 && Number.isFinite(axisLimit * 2);
  const chartHeight = Math.min(720, Math.max(320, chartItems.length * 34 + 100));
  const targetIds = [...new Set(chartItems.map((item) => item.candidateTest.target))];
  const richLabelStyles = {
    separator: { color: theme.palette.text.disabled },
    // One rich-text style per target so the label carries the same hue as the target tag/legend.
    ...Object.fromEntries(targetIds.map((target, index) => [`target${index}`, {
      color: targetTextColor(theme, target),
      fontWeight: fontWeights.emphasized,
      fontFamily: theme.typography.fontFamily,
    }])),
    // Thread policy stays neutral (as in the tags); the swatch is the same line drawing as ThreadPolicySwatch.
    mode: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily },
    singleSwatch: { width: SWATCH_WIDTH, height: SWATCH_HEIGHT, backgroundColor: { image: policySwatchImage(theme.palette.text.secondary, 'single') } },
    defaultSwatch: { width: SWATCH_WIDTH, height: SWATCH_HEIGHT, backgroundColor: { image: policySwatchImage(theme.palette.text.secondary, 'default') } },
    benchmark: {
      color: theme.palette.text.secondary,
      fontWeight: fontWeights.emphasized,
      fontFamily: theme.typography.fontFamily,
    },
  };

  const stateFor = (delta) => classifyDurationChange(delta, viewModel.tolerance);
  const colorFor = (delta) => {
    const state = stateFor(delta);
    if (state === 'slower') return theme.palette.error.main;
    if (state === 'faster') return theme.palette.success.main;
    return theme.palette.text.secondary;
  };
  const labelFor = (delta) => {
    const state = stateFor(delta);
    const indicator = state === 'slower' ? '↑' : state === 'faster' ? '↓' : '—';
    return `${indicator} ${formatPercent(delta)}`;
  };
  const categoryFor = (item) => {
    const clean = (value) => String(value).replace(/[{}|]/g, '');
    const target = clean(item.candidateTest.target);
    const mode = clean(item.candidateTest.mode);
    const swatch = item.candidateTest.mode === 'single' ? 'singleSwatch' : 'defaultSwatch';
    const benchmark = clean(item.candidateTest.name);
    const separator = compactChart ? '\n' : '{separator| · }';
    // Phones: the swatch alone carries the policy (as in the tags); the word needs width they lack.
    const policy = compactChart ? `{${swatch}|}` : `{${swatch}|} {mode|${mode}}`;
    return `{target${targetIds.indexOf(item.candidateTest.target)}|${target}} ${policy}${separator}{benchmark|${benchmark}}`;
  };

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (parameters) => {
        const point = (Array.isArray(parameters) ? parameters[0] : parameters).data;
        return [
        `<strong>${escapeHtml(point.comparison.candidateTest.name)}</strong>`,
        `${escapeHtml(point.comparison.candidateTest.target)} · ${escapeHtml(point.comparison.candidateTest.mode)} · ${escapeHtml(point.comparison.candidateTest.suite)}`,
        `Candidate ${formatDuration(point.comparison.candidateTest.durationSeconds)}`,
        `Baseline ${formatDuration(point.comparison.baselineTest.durationSeconds)}`,
        `Change ${formatPercent(point.comparison.delta)}`,
        `Commits ${escapeHtml(shortSha(candidate))} vs ${escapeHtml(shortSha(baseline))}`,
      ].join('<br/>');
      },
    },
    grid: compactChart
      ? { left: 150, right: 12, top: 18, bottom: 48 }
      : { left: 300, right: 88, top: 18, bottom: 48 },
    xAxis: {
      type: 'value',
      name: 'Duration change (%)',
      nameLocation: 'middle',
      nameGap: 32,
      min: -axisLimit,
      max: axisLimit,
      axisLabel: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily, formatter: '{value}%' },
      nameTextStyle: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily },
      axisLine: { show: true, lineStyle: { color: theme.palette.dashboard?.border ?? theme.palette.divider } },
      splitLine: { lineStyle: { color: theme.palette.divider, type: 'dashed' } },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      tooltip: {
        show: true,
        formatter: (params) => {
          if (!params.isTruncated?.()) return '';
          const item = chartItems[params.tickIndex]?.candidateTest;
          return item ? escapeHtml(`${item.target} · ${item.mode} · ${item.name}`) : '';
        },
      },
      data: chartItems.map(categoryFor),
      axisTick: { show: false },
      axisPointer: { show: true, type: 'shadow', shadowStyle: { color: alpha(theme.palette.action.hover, 0.6) } },
      axisLine: { show: false },
      axisLabel: {
        fontSize: fontSizes.caption,
        fontFamily: theme.typography.fontFamily,
        color: theme.palette.text.secondary,
        width: compactChart ? 132 : 270,
        overflow: 'truncate',
        formatter: (value) => value,
        rich: richLabelStyles,
      },
    },
    series: [{
      type: 'bar',
      barMaxWidth: 18,
      showBackground: true,
      backgroundStyle: { color: colorTokens[theme.palette.mode].barTrack, borderRadius: 6 },
      data: chartItems.map((item) => {
        const state = stateFor(item.delta);
        const performanceColor = state === 'faster' ? colorTokens[theme.palette.mode].successGraphic : state === 'slower' ? colorTokens[theme.palette.mode].errorGraphic : theme.palette.text.secondary;
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
              : colorFor(item.delta),
            fontSize: fontSizes.caption,
            fontFamily: theme.typography.fontFamily,
            fontWeight: fontWeights.emphasized,
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
        lineStyle: { color: theme.palette.text.secondary, width: 1 },
        data: [{ xAxis: 0 }],
      },
    }],
  };

  return (
    <Box data-testid="comparison-results" sx={{ display: 'grid', gap: '14px', minWidth: 0 }}>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'minmax(0, 1fr) auto minmax(0, 1fr)' }, alignItems: 'start', gap: '10px' }}>
        <RunSelector label="Candidate run" options={runOptions} value={candidate} onChange={onCandidateChange} />
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
            // Same surface and border as the run fields beside it; the blue icon marks it as the action.
            bgcolor: 'background.paper', color: 'text.primary', fontWeight: fontWeights.emphasized, borderColor: 'dashboard.controlBorder',
            '& .MuiButton-startIcon': { color: 'primary.main' },
            '&:hover': { bgcolor: 'action.hover', borderColor: 'text.secondary' },
            whiteSpace: 'nowrap',
            justifySelf: 'center',
            alignSelf: { xs: 'center', md: 'start' },
            mt: 0, height: { md: 37 }, py: 0, minHeight: { md: 0 },
          }}
        >
          Swap
        </Button>
        <RunSelector label="Baseline run" options={runOptions} value={baseline} onChange={onBaselineChange} />
      </Box>
      {!candidate && selectedCandidateId && <Alert severity="info">Selected candidate is unavailable · {selectedCandidateId}. Choose a published attempt; no different run has been substituted.</Alert>}
      {!baseline && selectedBaselineId && <Alert severity="info">Selected baseline is unavailable · {selectedBaselineId}. Choose a published attempt; no different run has been substituted.</Alert>}

      <SectionCard
        title="Performance change by benchmark"
        subtitle={(
          <>
            <Box component="span" sx={{ display: 'block' }}>Candidate vs baseline · Lower duration is faster</Box>
            <CommitComparison candidate={candidate} baseline={baseline} sx={{ mt: '2px' }} />
          </>
        )}
      >
        <ComparisonMetrics model={viewModel} candidate={candidate} baseline={baseline} />
        <Stack direction="row" sx={{ flexWrap: 'wrap', gap: '6px', mt: '10px' }}>
          <ChangeCountChip tone="success" label={`${hasRuns ? viewModel.counts.faster : '—'} faster`} />
          <ChangeCountChip label={`${hasRuns ? viewModel.counts.neutral : '—'} within ±${viewModel.tolerance}%`} />
          <ChangeCountChip tone="error" label={`${hasRuns ? viewModel.counts.slower : '—'} slower`} />
          {viewModel.counts.unavailable > 0 && <Chip size="small" variant="outlined" label={`${viewModel.counts.unavailable} percentage unavailable`} />}
        </Stack>
        <Typography variant="caption" sx={{ color: 'text.muted', display: 'block', mt: 1 }}>
          Totals use matched benchmarks with valid completed durations in both runs; bars and faster/slower/noise counts include only available percentage changes.
          Matching uses exact target + thread policy + workload identity, not environment equivalence. Exclusions are not zero-duration results. A zero baseline has no percentage ratio.
        </Typography>
        {chartItems.length > 0 && chartRangeAvailable ? (
          <Box sx={{ mx: { xs: '-10px', sm: '-4px' }, mt: '6px' }}>
            <Chart option={option} height={chartHeight} ariaLabel="Performance change by benchmark comparison chart" />
          </Box>
        ) : (
          <Typography sx={{ color: 'text.secondary', py: 8, textAlign: 'center' }}>{!hasRuns ? 'Select baseline and candidate runs to compare benchmark results.' : chartItems.length ? 'Chart unavailable: percentage changes exceed the supported chart range. Measured results remain in the results table below.' : viewModel.comparable.length ? 'No percentage changes available. Measured results remain in the results table below.' : 'No completed benchmark results are comparable between these runs.'}</Typography>
        )}
        {viewModel.comparable.length > 0 && (
          <Disclosure title={`View comparable results (${viewModel.comparable.length})`}>
            <ScrollTableContainer tabIndex={0} role="region" aria-label="Comparable results scroll area" sx={{ maxHeight: 250 }}>
              <Table size="small" stickyHeader aria-label="Comparable benchmark results" sx={{ ...tableStyles.normal, minWidth: 540 }}>
                <TableHead><TableRow>
                  <TableCell>Benchmark / target / thread policy</TableCell><TableCell align="right">Baseline</TableCell><TableCell align="right">Candidate</TableCell><TableCell align="right">Change</TableCell>
                </TableRow></TableHead>
                <TableBody>{viewModel.comparable.map((item) => (
                  <TableRow key={item.candidateTest.testId} data-different={['slower', 'faster'].includes(stateFor(item.delta))} sx={['slower', 'faster'].includes(stateFor(item.delta)) ? changedRowSx : undefined}>
                    <TableCell sx={{ overflowWrap: 'anywhere', maxWidth: 320 }}>{item.candidateTest.name} · {item.candidateTest.target} · {item.candidateTest.mode}</TableCell>
                    <TableCell sx={numericTableCell}>{formatDuration(item.baselineTest.durationSeconds)}</TableCell>
                    <TableCell sx={numericTableCell}>{formatDuration(item.candidateTest.durationSeconds)}</TableCell>
                    <TableCell sx={{ ...numericTableCell, color: colorFor(item.delta), whiteSpace: 'nowrap' }}>{formatPercent(item.delta)} · {Number.isFinite(item.delta) ? stateFor(item.delta) === 'neutral' ? `within ±${viewModel.tolerance}%` : stateFor(item.delta) : 'percentage unavailable'}</TableCell>
                  </TableRow>
                ))}</TableBody>
              </Table>
            </ScrollTableContainer>
          </Disclosure>
        )}
      </SectionCard>

      <SectionCard
        title="Compared run information"
        subtitle="Completeness and catalog availability reflect the active target, suite and thread policy filters"
      >
        <RunMetadataDiff baseline={baseline} candidate={candidate} filters={filters} />
      </SectionCard>

      <SectionCard
        title="Tests not included in comparison"
        subtitle="Catalog availability and benchmark-result status for the active target, suite and thread policy filters"
      >
        <ExcludedTestsTable comparisons={viewModel.notComparable} comparableCount={viewModel.comparable.length} baseline={baseline} candidate={candidate} />
      </SectionCard>
    </Box>
  );
}
