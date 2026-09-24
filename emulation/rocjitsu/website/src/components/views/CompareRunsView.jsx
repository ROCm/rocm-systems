import {
  Box,
  Button,
  Chip,
  Divider,
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
import ArrowDownwardRoundedIcon from '@mui/icons-material/ArrowDownwardRounded';
import ArrowUpwardRoundedIcon from '@mui/icons-material/ArrowUpwardRounded';
import RemoveRoundedIcon from '@mui/icons-material/RemoveRounded';
import SwapVertRoundedIcon from '@mui/icons-material/SwapVertRounded';
import RunSelector from '../compare/RunSelector';
import Chart from '../shared/Chart';
import DetailItem from '../shared/DetailItem';
import { DetailGrid, DetailSectionHeading } from '../shared/DetailLayout';
import SectionCard from '../shared/SectionCard';
import StatusChip from '../shared/StatusChip';
import { previousCompletedRunForFilters, selectRunComparison } from '../../data/selectors';
import { commitTimestampFor } from '../../data/runOrdering';
import { provenanceDetails } from '../../data/provenance';
import {
  escapeHtml,
  formatDuration,
  formatFullDate,
  formatPercent,
  shortSha,
} from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';
import CommitComparison from '../shared/CommitComparison';

const NOISE_TOLERANCE = 3;

function SummaryStat({ label, value }) {
  return (
    <Box sx={{ p: 1.35, border: 1, borderColor: 'divider', borderRadius: 2, bgcolor: 'action.hover' }}>
      <Typography variant="caption" sx={{ color: 'text.secondary' }}>{label}</Typography>
      <Typography sx={{ fontSize: 20, fontWeight: 770, lineHeight: 1.2, mt: 0.3 }}>{value}</Typography>
    </Box>
  );
}

function AggregateChange({ value, candidate, baseline }) {
  const state = classifyDurationChange(value, NOISE_TOLERANCE);
  const tone = changeTone(state);
  const color = tone === 'neutral' ? 'text.secondary' : `${tone}.main`;
  return (
    <Box sx={{ p: 1.35, border: 1, borderColor: 'divider', borderRadius: 2, bgcolor: 'action.hover' }}>
      <Typography variant="caption" sx={{ color: 'text.secondary' }}>Aggregate change</Typography>
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

function HighlightedValue({ different, children }) {
  return (
    <Box component="span" sx={different ? { color: 'warning.main', fontWeight: 750 } : undefined}>
      {children}
    </Box>
  );
}

function RunInformation({ label, run, otherRun, filters, accentColor }) {
  if (!run) return null;
  const selectedTests = run.tests.filter((test) => (
    filters.targets.includes(test.target) && filters.suites.includes(test.suite)
  ));
  const otherSelectedTests = (otherRun?.tests ?? []).filter((test) => (
    filters.targets.includes(test.target) && filters.suites.includes(test.suite)
  ));
  const completed = selectedTests.filter((test) => (
    test.status === 'completed' && Number.isFinite(test.durationSeconds)
  )).length;
  const otherCompleted = otherSelectedTests.filter((test) => (
    test.status === 'completed' && Number.isFinite(test.durationSeconds)
  )).length;
  const complete = selectedTests.length > 0 && completed === selectedTests.length;
  const environment = provenanceDetails(run.provenance);
  const otherEnvironment = new Map(
    provenanceDetails(otherRun?.provenance).map((detail) => [detail.key, detail.value]),
  );
  const coverage = `${completed}/${selectedTests.length} completed`;
  const otherCoverage = `${otherCompleted}/${otherSelectedTests.length} completed`;

  return (
    <Box
      data-testid={`${label.toLowerCase()}-run-information`}
      sx={{
        minWidth: 0,
        p: { xs: 1.5, sm: 2 },
        border: 1,
        borderTop: 3,
        borderColor: 'divider',
        borderTopColor: accentColor,
        borderRadius: 2,
        bgcolor: 'background.paper',
      }}
    >
      <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', gap: 1, mb: 1.5 }}>
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="overline" sx={{ color: 'text.secondary' }}>{label} run</Typography>
          <Typography component="div" sx={{ fontFamily: 'monospace', fontWeight: 780 }}>
            {shortSha(run)}
          </Typography>
          {run.provenance?.commitMessage && (
            <Typography variant="caption" sx={{ color: 'text.secondary' }} noWrap title={run.provenance.commitMessage}>
              {run.provenance.commitMessage}
            </Typography>
          )}
        </Box>
        <Chip
          size="small"
          color={complete ? 'success' : 'error'}
          variant="outlined"
          label={complete ? 'Completed' : 'Not completed'}
        />
      </Stack>
      <DetailGrid>
        <DetailItem label="Coverage">
          <HighlightedValue different={coverage !== otherCoverage}>{coverage}</HighlightedValue>
        </DetailItem>
        <DetailItem label="Test catalog">
          <HighlightedValue different={run.catalogId !== otherRun?.catalogId}>{run.catalogId}</HighlightedValue>
        </DetailItem>
        <DetailItem label="Commit time">{formatFullDate(commitTimestampFor(run))}</DetailItem>
        <DetailItem label="Run time">{formatFullDate(run.timestamp)}</DetailItem>
        <DetailItem label="Machine">
          <HighlightedValue different={run.machineId !== otherRun?.machineId}>{run.machineId}</HighlightedValue>
        </DetailItem>
      </DetailGrid>
      <Divider sx={{ my: 1.75 }} />
      <DetailSectionHeading>Environment</DetailSectionHeading>
      {environment.length > 0 ? (
        <DetailGrid>
          {environment.map((detail) => (
            <DetailItem key={detail.key} label={detail.label}>
              <HighlightedValue different={detail.value !== otherEnvironment.get(detail.key)}>
                {detail.value}
              </HighlightedValue>
            </DetailItem>
          ))}
        </DetailGrid>
      ) : (
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>No environment details provided.</Typography>
      )}
    </Box>
  );
}

function TestAvailability({ test }) {
  return test
    ? <StatusChip status={test.status} />
    : <Chip size="small" variant="outlined" label="Unavailable in catalog" />;
}

function ExcludedTestsTable({ comparisons }) {
  if (comparisons.length === 0) {
    return (
      <Typography sx={{ color: 'text.secondary', py: 3, textAlign: 'center' }}>
        Every selected benchmark has completed data in both runs.
      </Typography>
    );
  }

  return (
    <TableContainer sx={{ border: 1, borderColor: 'divider', borderRadius: 2 }}>
      <Table size="small" sx={{ minWidth: 760 }}>
        <TableHead>
          <TableRow>
            <TableCell>Benchmark</TableCell>
            <TableCell>Target</TableCell>
            <TableCell>Suite</TableCell>
            <TableCell>Candidate</TableCell>
            <TableCell>Baseline</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {comparisons.map((comparison) => {
            const benchmark = comparison.candidateTest ?? comparison.baselineTest;
            return (
              <TableRow key={benchmark.testId}>
                <TableCell sx={{ fontWeight: 700 }}>{benchmark.name}</TableCell>
                <TableCell>{benchmark.target}</TableCell>
                <TableCell>{benchmark.suite}</TableCell>
                <TableCell><TestAvailability test={comparison.candidateTest} /></TableCell>
                <TableCell><TestAvailability test={comparison.baselineTest} /></TableCell>
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
}) {
  const theme = useTheme();
  const compactChart = useMediaQuery(theme.breakpoints.down('sm'));
  const selectedCandidate = data.runs.find((run) => run.runId === selectedCandidateId);
  const candidate = selectedCandidate ?? data.latestRun;
  const defaultBaseline = selectedCandidate
    ? previousCompletedRunForFilters(data.runs, candidate, filters)
    : data.runs.at(-2) ?? null;
  const baseline = data.runs.find((run) => run.runId === selectedBaselineId) ?? defaultBaseline;
  const viewModel = selectRunComparison(candidate, baseline, filters, NOISE_TOLERANCE);
  const hasRuns = Boolean(candidate);
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
    const target = item.candidateTest.target.replace(/[{}|]/g, '');
    const benchmark = item.candidateTest.name.replace(/[{}|]/g, '');
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
        `<strong>${escapeHtml(point.comparison.candidateTest.name)}</strong>`,
        `${escapeHtml(point.comparison.candidateTest.target)} · ${escapeHtml(point.comparison.candidateTest.suite)}`,
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
          <SummaryStat label="Candidate total" value={hasRuns ? formatDuration(viewModel.candidateDuration) : '—'} />
          <SummaryStat label="Baseline total" value={hasRuns ? formatDuration(viewModel.baselineDuration) : '—'} />
          <SummaryStat label="Comparable" value={hasRuns ? viewModel.comparable.length : '—'} />
          <SummaryStat label="Not comparable" value={hasRuns ? viewModel.notComparable.length : '—'} />
        </Box>
        <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 0.75, mt: 1.25 }}>
          <Chip size="small" color="success" variant="outlined" label={`${hasRuns ? viewModel.counts.faster : '—'} faster`} />
          <Chip size="small" variant="outlined" label={`${hasRuns ? viewModel.counts.neutral : '—'} within ±${NOISE_TOLERANCE}%`} />
          <Chip size="small" color="error" variant="outlined" label={`${hasRuns ? viewModel.counts.slower : '—'} slower`} />
        </Stack>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 1 }}>
          Only benchmarks with valid completed durations in both runs are compared.
        </Typography>
        {viewModel.comparable.length > 0 ? (
          <Box sx={{ mx: { xs: -1.25, sm: -0.5 }, mt: 0.75 }}>
            <Chart option={option} height={chartHeight} ariaLabel="Performance change by benchmark comparison chart" />
          </Box>
        ) : (
          <Typography sx={{ color: 'text.secondary', py: 8, textAlign: 'center' }}>No completed benchmark results are comparable between these runs.</Typography>
        )}
      </SectionCard>

      <SectionCard
        title="Compared Run Information"
        subtitle="Completeness and catalog availability reflect the active target and suite filters"
      >
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, minmax(0, 1fr))' }, gap: 1.25 }}>
          <RunInformation
            label="Candidate"
            run={candidate}
            otherRun={baseline}
            filters={filters}
            accentColor="primary.main"
          />
          <RunInformation
            label="Baseline"
            run={baseline}
            otherRun={candidate}
            filters={filters}
            accentColor="secondary.main"
          />
        </Box>
      </SectionCard>

      <SectionCard
        title="Tests Not Included in Comparison"
        subtitle="Catalog availability and benchmark-result status for the active target and suite filters"
      >
        <ExcludedTestsTable comparisons={viewModel.notComparable} />
      </SectionCard>
    </Box>
  );
}
