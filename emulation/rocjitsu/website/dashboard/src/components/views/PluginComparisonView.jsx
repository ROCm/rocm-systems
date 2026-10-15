import { useMemo, useState } from 'react';
import {
  Alert,
  Box,
  ButtonBase,
  Chip,
  MenuItem,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import BugReportRoundedIcon from '@mui/icons-material/BugReportRounded';
import ErrorRoundedIcon from '@mui/icons-material/ErrorRounded';
import ExtensionRoundedIcon from '@mui/icons-material/ExtensionRounded';
import MemoryRoundedIcon from '@mui/icons-material/MemoryRounded';
import TimerOffRoundedIcon from '@mui/icons-material/TimerOffRounded';
import { alpha } from '@mui/material/styles';
import Chart from '../shared/Chart';
import SectionCard from '../shared/SectionCard';
import StatusChip from '../shared/StatusChip';
import BenchmarkResultDialog from '../benchmarks/BenchmarkResultDialog';
import {
  PLUGIN_NOISE_TOLERANCE,
  selectPluginComparison,
  selectPluginComparisonGroups,
} from '../../data/pluginComparison';
import {
  escapeHtml,
  formatDuration,
  formatFullDate,
  formatPercent,
  shortSha,
} from '../../utils/formatters';
import { detailActionStyles } from '../../theme/styles';

const PLUGIN_COLORS = {
  vanilla: '#16A34A',
  asan: '#D97706',
  tsan: '#8B5CF6',
  ubsan: '#0891B2',
};
const TARGET_COLORS = { light: '#2563EB', dark: '#60A5FA' };
const FALLBACK_PLUGIN_COLORS = ['#DB2777', '#CA8A04', '#DC2626', '#4F46E5'];

function pluginColor(pluginId) {
  if (PLUGIN_COLORS[pluginId]) return PLUGIN_COLORS[pluginId];
  const hash = [...pluginId].reduce((value, character) => ((value * 31) + character.charCodeAt(0)) >>> 0, 0);
  return FALLBACK_PLUGIN_COLORS[hash % FALLBACK_PLUGIN_COLORS.length];
}

function comparisonPluginNames(group) {
  const plugins = group.runs
    .filter((run) => run.plugin.id !== 'vanilla')
    .map((run) => run.plugin.name);
  return plugins.length > 0 ? plugins.join(', ') : 'Vanilla';
}

function pluginErrorSummary(error, pluginName) {
  const prefix = `${pluginName}:`;
  const summary = error.toLowerCase().startsWith(prefix.toLowerCase())
    ? error.slice(prefix.length).trim()
    : error;
  return summary ? `${summary[0].toUpperCase()}${summary.slice(1)}` : error;
}

function SummaryCard({ summary, baseline, target }) {
  const { run } = summary;
  const color = pluginColor(run.plugin.id);
  return (
    <Paper
      data-testid={`plugin-summary-${target}-${run.plugin.id}`}
      variant="outlined"
      sx={(theme) => ({
        p: 1.5,
        minWidth: 0,
        borderRadius: 2.25,
        borderTopWidth: 3,
        borderTopColor: color,
        backgroundColor: theme.palette.mode === 'dark' ? '#18202F' : '#FCFCFD',
        backgroundImage: 'none',
        boxShadow: 'none',
      })}
    >
      <Stack direction="row" sx={{ alignItems: 'flex-start', justifyContent: 'space-between', gap: 1 }}>
        <Box sx={{ minWidth: 0 }}>
          <Typography fontWeight={780} color={color} noWrap>{run.plugin.name}</Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {run.plugin.version ?? run.plugin.id}
          </Typography>
        </Box>
        {baseline && <Chip size="small" variant="outlined" label="Baseline" sx={{ color, borderColor: color }} />}
      </Stack>
      <Typography
        sx={{ mt: 1.2, fontSize: 24, lineHeight: 1.1, fontWeight: 820, color }}
      >
        {baseline ? 'Baseline' : Number.isFinite(summary.overhead) ? `${formatPercent(summary.overhead)}${summary.estimated ? '*' : ''}` : '—'}
      </Typography>
      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
        {baseline
          ? run.plugin.id === 'vanilla' ? 'Uninstrumented reference' : 'Selected baseline'
          : summary.estimated
            ? `Estimated for all ${summary.total} tests from ${summary.comparable} passed pairs`
            : Number.isFinite(summary.overhead)
              ? 'Geometric-mean runtime overhead'
              : 'No comparable completed tests'}
      </Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 0.75, mt: 1.3 }}>
        <Box>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>Coverage</Typography>
          <Typography variant="body2" fontWeight={720}>{summary.completed}/{summary.total}</Typography>
        </Box>
        <Box>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>Total duration</Typography>
          <Typography variant="body2" fontWeight={720}>{formatDuration(summary.duration)}</Typography>
        </Box>
      </Box>
      {!baseline && (
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 1 }}>
          {summary.counts.faster} lower · {summary.counts.neutral} within ±{PLUGIN_NOISE_TOLERANCE}% · {summary.counts.slower} overhead
        </Typography>
      )}
      {(summary.failed > 0 || summary.timeout > 0) && (
        <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 0.65, mt: 0.75 }}>
          {summary.failed > 0 && (
            <Chip
              size="small"
              variant="outlined"
              color="error"
              icon={<ErrorRoundedIcon />}
              label={`${summary.failed} failed`}
              sx={(theme) => ({
                height: 23,
                bgcolor: alpha(theme.palette.error.main, 0.07),
                '& .MuiChip-label': { px: 0.8, fontSize: '0.68rem', fontWeight: 720 },
                '& .MuiChip-icon': { ml: 0.65, fontSize: 14 },
              })}
            />
          )}
          {summary.timeout > 0 && (
            <Chip
              size="small"
              variant="outlined"
              color="warning"
              icon={<TimerOffRoundedIcon />}
              label={`${summary.timeout} timed out`}
              sx={(theme) => ({
                height: 23,
                bgcolor: alpha(theme.palette.warning.main, 0.07),
                '& .MuiChip-label': { px: 0.8, fontSize: '0.68rem', fontWeight: 720 },
                '& .MuiChip-icon': { ml: 0.65, fontSize: 14 },
              })}
            />
          )}
        </Stack>
      )}
    </Paper>
  );
}

function ResultCell({ value, baseline, onOpen }) {
  if (!value?.result) return <Typography sx={{ color: 'text.secondary' }}>Missing</Typography>;
  return (
    <ButtonBase
      onClick={() => onOpen({ run: value.run, test: value.result })}
      aria-label={`Open ${value.run.plugin.name} result for ${value.result.name}`}
      sx={{ ...detailActionStyles, width: '100%' }}
    >
      <Box>
        {value.result.status === 'completed' ? (
          <Typography variant="body2" fontWeight={720}>{formatDuration(value.result.durationSeconds)}</Typography>
        ) : (
          <StatusChip status={value.result.status} />
        )}
        <Typography
          variant="caption"
          sx={{
            color: baseline || !Number.isFinite(value.delta) ? 'text.secondary' : value.delta > 0 ? 'warning.main' : 'success.main',
            display: 'block',
            mt: 0.15,
          }}
        >
          {baseline ? 'Baseline' : Number.isFinite(value.delta) ? `${formatPercent(value.delta)} overhead` : 'Not comparable'}
        </Typography>
      </Box>
    </ButtonBase>
  );
}

function TargetComparison({ group, target, suites, baselinePluginId, onOpen }) {
  const theme = useTheme();
  const compact = useMediaQuery(theme.breakpoints.down('sm'));
  const viewModel = useMemo(
    () => selectPluginComparison(group, target, suites, baselinePluginId),
    [group, target, suites, baselinePluginId],
  );
  const comparisonRuns = viewModel.pluginRuns.filter((run) => run.runId !== viewModel.baselineRun?.runId);
  const baselineOnly = comparisonRuns.length === 0;
  const allDeltas = viewModel.rows.flatMap((row) => row.values
    .filter((value) => value.run.runId !== viewModel.baselineRun?.runId && Number.isFinite(value.delta))
    .map((value) => value.delta));
  const smallestDelta = Math.min(...allDeltas, 0);
  const largestDelta = Math.max(...allDeltas, 0);
  const axisMinimum = smallestDelta < 0 ? Math.floor((smallestDelta * 1.15) / 10) * 10 : 0;
  const axisMaximum = largestDelta > 0 ? Math.max(10, Math.ceil((largestDelta * 1.15) / 10) * 10) : 0;
  const chartOption = {
    tooltip: {
      trigger: 'item',
      formatter: ({ data: point }) => point?.value == null ? '' : [
        `<strong>${escapeHtml(point.test.name)}</strong>`,
        `${escapeHtml(point.run.plugin.name)} on ${escapeHtml(target)}`,
        `Duration ${formatDuration(point.result.durationSeconds)}`,
        `Runtime overhead ${formatPercent(point.delta)}`,
      ].join('<br/>'),
    },
    legend: {
      top: 0,
      textStyle: { color: theme.palette.text.secondary },
    },
    grid: compact
      ? { left: 112, right: 14, top: 46, bottom: 44 }
      : { left: 235, right: 54, top: 46, bottom: 44 },
    xAxis: {
      type: 'value',
      show: !baselineOnly,
      name: `Runtime overhead vs ${viewModel.baselineRun?.plugin.name ?? 'baseline'} (%)`,
      nameLocation: 'middle',
      nameGap: 30,
      min: axisMinimum,
      max: axisMaximum,
      axisLabel: { formatter: '{value}%', color: theme.palette.text.secondary },
      splitLine: { lineStyle: { color: theme.palette.divider, type: 'dashed' } },
    },
    yAxis: {
      type: 'category',
      show: !baselineOnly,
      inverse: true,
      data: viewModel.rows.map((row) => compact ? row.test.name : `${row.test.suite} · ${row.test.name}`),
      axisTick: { show: false },
      axisLine: { show: false },
      axisLabel: {
        color: theme.palette.text.secondary,
        width: compact ? 96 : 215,
        overflow: 'truncate',
        fontSize: compact ? 10 : 11,
      },
    },
    series: comparisonRuns.map((run) => ({
      name: run.plugin.name,
      type: 'bar',
      barMaxWidth: 18,
      itemStyle: { color: pluginColor(run.plugin.id) },
      data: viewModel.rows.map((row) => {
        const value = row.values.find((candidate) => candidate.run.runId === run.runId);
        return value?.comparable ? {
          value: Number(value.delta.toFixed(3)),
          ...value,
          test: row.test,
          itemStyle: {
            color: pluginColor(run.plugin.id),
            borderRadius: value.delta >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4],
          },
        } : null;
      }),
      label: {
        show: !compact,
        position: 'right',
        color: theme.palette.text.secondary,
        formatter: ({ value }) => formatPercent(value),
      },
      emphasis: { focus: 'series' },
      markLine: {
        silent: true,
        symbol: 'none',
        label: { show: false },
        lineStyle: { color: theme.palette.text.primary, width: 1.2 },
        data: [{ xAxis: 0 }],
      },
    })),
  };

  return (
    <Box data-testid={`plugin-target-${target}`} component="section" aria-labelledby={`plugin-target-heading-${target}`} sx={{ display: 'grid', gap: 1.25 }}>
      <Paper
        variant="outlined"
        sx={(theme) => {
          const targetColor = TARGET_COLORS[theme.palette.mode];
          return {
            '--target-color': targetColor,
            mt: 0.5,
            px: { xs: 1.2, sm: 1.5 },
            py: 1.1,
            borderRadius: 2.25,
            borderColor: alpha(targetColor, theme.palette.mode === 'dark' ? 0.44 : 0.24),
            borderLeftWidth: 4,
            borderLeftColor: targetColor,
            bgcolor: theme.palette.mode === 'dark' ? '#151C2B' : '#FFFFFF',
            backgroundImage: 'none',
            boxShadow: 'none',
          };
        }}
      >
        <Stack direction="row" sx={{ alignItems: 'center', gap: { xs: 1.1, sm: 1.4 } }}>
          <Box
            sx={{
              width: 38,
              height: 38,
              flex: '0 0 38px',
              display: 'grid',
              placeItems: 'center',
              borderRadius: 1.5,
              bgcolor: 'var(--target-color)',
              color: '#FFFFFF',
            }}
          >
            <MemoryRoundedIcon sx={{ fontSize: 21 }} />
          </Box>
          <Box sx={{ minWidth: { xs: 82, sm: 120 } }}>
            <Typography variant="overline" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.58rem', lineHeight: 1.15 }}>
              Target
            </Typography>
            <Typography
              id={`plugin-target-heading-${target}`}
              component="h2"
              variant="h2"
              sx={{ mt: 0.15, color: 'var(--target-color)', fontFamily: 'monospace', fontSize: '1.08rem', fontWeight: 820, letterSpacing: '0.01em' }}
            >
              {target}
            </Typography>
          </Box>
          <Box aria-hidden="true" sx={{ width: '1px', height: 32, bgcolor: 'divider' }} />
          <Box sx={{ minWidth: 48 }}>
            <Typography variant="overline" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.58rem', lineHeight: 1.15 }}>
              Plugins
            </Typography>
            <Typography variant="body2" sx={{ mt: 0.2, color: 'text.primary', fontWeight: 800 }}>
              {viewModel.pluginRuns.length}
            </Typography>
          </Box>
          <Box sx={{ minWidth: 40 }}>
            <Typography variant="overline" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.58rem', lineHeight: 1.15 }}>
              Tests
            </Typography>
            <Typography variant="body2" sx={{ mt: 0.2, color: 'text.primary', fontWeight: 800 }}>
              {viewModel.rows.length}
            </Typography>
          </Box>
        </Stack>
      </Paper>

      {viewModel.rows.length === 0 ? (
        <Alert severity="info">No tests match the selected global Suite filters for {target}.</Alert>
      ) : (
        <>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))', lg: `repeat(${viewModel.summaries.length + Number(baselineOnly)}, minmax(0, 1fr))` }, gap: 1.25 }}>
            {viewModel.summaries.map((summary) => (
              <SummaryCard
                key={summary.run.runId}
                summary={summary}
                target={target}
                baseline={summary.run.runId === viewModel.baselineRun?.runId}
              />
            ))}
            {baselineOnly && (
              <Paper variant="outlined" sx={{ p: 1.5, borderRadius: 2.25, borderTopWidth: 3, borderTopColor: 'text.disabled' }}>
                <Typography fontWeight={780}>Comparison plugin</Typography>
                <Typography sx={{ mt: 1.2, fontSize: 24, lineHeight: 1.1, fontWeight: 820 }}>—</Typography>
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 0.75, mt: 1.3 }}>
                  <Box><Typography variant="caption" sx={{ color: 'text.secondary' }}>Coverage</Typography><Typography variant="body2" fontWeight={720}>—</Typography></Box>
                  <Box><Typography variant="caption" sx={{ color: 'text.secondary' }}>Total duration</Typography><Typography variant="body2" fontWeight={720}>—</Typography></Box>
                </Box>
              </Paper>
            )}
          </Box>

          <SectionCard
            title="Per-Test Runtime Overhead"
            subtitle={`Lower is better · Values within ±${PLUGIN_NOISE_TOLERANCE}% are treated as measurement noise`}
          >
            <Chart
              option={chartOption}
              height={Math.min(650, Math.max(340, viewModel.rows.length * comparisonRuns.length * 24 + 120))}
              ariaLabel={`Plugin runtime overhead for ${target}`}
              ariaDescribedBy={`plugin-chart-help-${target}`}
              onEvents={{ click: ({ data: point }) => point?.result && onOpen({ run: point.run, test: point.result }) }}
            />
            <Typography id={`plugin-chart-help-${target}`} variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 1 }}>
              Every point in this chart is also available as a keyboard-operable button in the Test-by-Plugin Results table below.
            </Typography>
            {viewModel.summaries.some((summary) => summary.estimated) && (
              <Alert severity="warning" variant="outlined" sx={{ mt: 1 }}>
                * Estimated for the full selected set: the geometric-mean overhead measured from passed plugin/baseline pairs is assumed for failed, timed-out, missing, or baseline-incomplete results.
              </Alert>
            )}
          </SectionCard>

          <SectionCard title="Test-by-Plugin Results" subtitle="Select any result to inspect its problem, plugin, environment, and provenance">
            <TableContainer sx={{ border: 1, borderColor: 'divider', borderRadius: 2.25 }}>
              <Table size="small" sx={{ minWidth: 680 }}>
                <TableHead>
                  <TableRow>
                    <TableCell>Test</TableCell>
                    {viewModel.pluginRuns.map((run) => (
                      <TableCell key={run.runId}>{run.plugin.name}</TableCell>
                    ))}
                    {baselineOnly && <TableCell>Plugin</TableCell>}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {viewModel.rows.map((row) => (
                    <TableRow key={row.test.logicalTestId} hover>
                      <TableCell sx={{ minWidth: 220 }}>
                        <Typography variant="body2" fontWeight={720}>{row.test.name}</Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>{row.test.suite}</Typography>
                      </TableCell>
                      {row.values.map((value) => (
                        <TableCell key={value.run.runId} sx={{ minWidth: 145 }}>
                          <ResultCell
                            value={value}
                            baseline={value.run.runId === viewModel.baselineRun?.runId}
                            onOpen={onOpen}
                          />
                        </TableCell>
                      ))}
                      {baselineOnly && <TableCell sx={{ color: 'text.secondary' }}>—</TableCell>}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </SectionCard>

          <SectionCard
            title="Plugin Errors"
            subtitle="Failure and timeout diagnostics for this target"
            action={<Chip size="small" icon={<BugReportRoundedIcon />} label={viewModel.errors.length} color={viewModel.errors.length ? 'warning' : 'default'} sx={{ alignSelf: 'flex-start' }} />}
          >
            {viewModel.errors.length === 0 ? (
              <Alert severity="success">No plugin failures or timeouts were reported.</Alert>
            ) : (
              <Stack spacing={1}>
                {viewModel.errors.map(({ run, test, error }, index) => (
                  <Box
                    key={`${run.runId}:${test.testId}:${index}`}
                    sx={{
                      p: 1.5,
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 2,
                      bgcolor: 'action.hover',
                    }}
                  >
                    <Stack direction={{ xs: 'column', sm: 'row' }} sx={{ justifyContent: 'space-between', gap: 1 }}>
                      <Box>
                        <Typography variant="body2" fontWeight={750}>{pluginErrorSummary(error, run.plugin.name)}</Typography>
                        <Stack direction="row" sx={{ alignItems: 'center', flexWrap: 'wrap', gap: 0.75, mt: 0.65 }}>
                          <Chip
                            size="small"
                            variant="outlined"
                            label={run.plugin.name}
                            sx={{
                              height: 22,
                              color: pluginColor(run.plugin.id),
                              borderColor: pluginColor(run.plugin.id),
                              bgcolor: alpha(pluginColor(run.plugin.id), 0.06),
                            }}
                          />
                          <Typography variant="caption" sx={{ color: 'text.secondary' }}>{test.name}</Typography>
                        </Stack>
                      </Box>
                      <StatusChip status={test.status} />
                    </Stack>
                  </Box>
                ))}
              </Stack>
            )}
          </SectionCard>
        </>
      )}
    </Box>
  );
}

function PluginComparisonNotice() {
  return (
    <Paper
      data-testid="plugin-comparison-empty"
      variant="outlined"
      sx={{ p: { xs: 2, sm: 2.5 }, borderRadius: 3, bgcolor: 'action.hover' }}
    >
      <Stack direction="row" sx={{ alignItems: 'flex-start', gap: 1.5 }}>
        <Box sx={{ width: 40, height: 40, flex: '0 0 40px', display: 'grid', placeItems: 'center', borderRadius: 2, bgcolor: 'background.paper', color: 'primary.main' }}>
          <ExtensionRoundedIcon />
        </Box>
        <Box>
          <Typography variant="h3">No sanitizer comparison runs available</Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
            No vanilla baseline or sanitizer comparison runs are available. To enable comparison, publish a vanilla run and at least one sanitizer run for the same commit.
          </Typography>
        </Box>
      </Stack>
    </Paper>
  );
}

function EmptyPluginComparison() {
  const emptyChartOption = {
    xAxis: { show: false },
    yAxis: { show: false },
    series: [],
  };

  return (
    <Box data-testid="plugin-comparison" sx={{ display: 'grid', gap: 1.75 }}>
      <PluginComparisonNotice />
      <SectionCard
        title="Plugin Comparison"
        subtitle="Runtime overhead and test health for controlled Vanilla and sanitizer executions"
      >
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1.5fr 1fr' }, gap: 1.25 }}>
          <TextField disabled size="small" label="Comparison experiment" value="" />
          <TextField disabled size="small" label="Baseline plugin" value="" />
        </Box>
      </SectionCard>

      <Paper variant="outlined" sx={{ px: 1.5, py: 1.25, borderRadius: 2.25, borderLeftWidth: 4, borderLeftColor: 'primary.main' }}>
        <Stack direction="row" sx={{ alignItems: 'center', gap: 2 }}>
          <MemoryRoundedIcon color="primary" />
          {[
            ['Target', '—'],
            ['Plugins', '—'],
            ['Tests', '—'],
          ].map(([label, value]) => (
            <Box key={label}>
              <Typography variant="overline" sx={{ color: 'text.secondary' }}>{label}</Typography>
              <Typography variant="body2" fontWeight={800}>{value}</Typography>
            </Box>
          ))}
        </Stack>
      </Paper>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))' }, gap: 1.25 }}>
        {['Baseline plugin', 'Comparison plugin'].map((label) => (
          <Paper key={label} variant="outlined" sx={{ p: 1.5, borderRadius: 2.25, borderTopWidth: 3, borderTopColor: 'text.disabled' }}>
            <Typography fontWeight={780}>{label}</Typography>
            <Typography sx={{ mt: 1.2, fontSize: 24, fontWeight: 820 }}>—</Typography>
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 0.75, mt: 1.3 }}>
              <Box><Typography variant="caption" sx={{ color: 'text.secondary' }}>Coverage</Typography><Typography>—</Typography></Box>
              <Box><Typography variant="caption" sx={{ color: 'text.secondary' }}>Total duration</Typography><Typography>—</Typography></Box>
            </Box>
          </Paper>
        ))}
      </Box>

      <SectionCard title="Per-Test Runtime Overhead" subtitle={`Lower is better · Values within ±${PLUGIN_NOISE_TOLERANCE}% are treated as measurement noise`}>
        <Chart option={emptyChartOption} height={340} ariaLabel="Empty plugin runtime overhead chart" />
      </SectionCard>

      <SectionCard title="Test-by-Plugin Results" subtitle="Select any result to inspect its problem, plugin, environment, and provenance">
        <TableContainer sx={{ border: 1, borderColor: 'divider', borderRadius: 2.25 }}>
          <Table size="small">
            <TableHead><TableRow><TableCell>Test</TableCell><TableCell>Baseline</TableCell><TableCell>Plugin</TableCell></TableRow></TableHead>
            <TableBody><TableRow>{['test', 'baseline', 'plugin'].map((column) => <TableCell key={column} sx={{ py: 4, color: 'text.secondary' }}>—</TableCell>)}</TableRow></TableBody>
          </Table>
        </TableContainer>
      </SectionCard>

      <SectionCard
        title="Plugin Errors"
        subtitle="Failure and timeout diagnostics for this target"
        action={<Chip size="small" icon={<BugReportRoundedIcon />} label="—" />}
      >
        <Typography sx={{ color: 'text.secondary' }}>—</Typography>
      </SectionCard>
    </Box>
  );
}

export default function PluginComparisonView({ data, filters }) {
  const comparisonGroups = useMemo(() => selectPluginComparisonGroups(data), [data]);
  const baselineOnlyGroup = useMemo(() => {
    if (comparisonGroups.length > 0 || data.runs.length === 0) return null;
    const run = data.latestCommitRun ?? data.latestRun ?? data.runs.at(-1);
    return {
      comparisonId: run.comparisonId,
      runs: [run],
      referenceRun: run,
      targets: run.targets,
    };
  }, [comparisonGroups, data]);
  const groups = baselineOnlyGroup ? [baselineOnlyGroup] : comparisonGroups;
  const [selectedGroupId, setSelectedGroupId] = useState(groups.at(-1)?.comparisonId ?? '');
  const selectedGroup = groups.find((group) => group.comparisonId === selectedGroupId) ?? groups.at(-1);
  const targets = selectedGroup?.targets.filter((target) => filters.targets.includes(target)) ?? [];
  const [requestedBaseline, setRequestedBaseline] = useState('vanilla');
  const baselinePluginId = selectedGroup?.runs.some((run) => run.plugin.id === requestedBaseline)
    ? requestedBaseline
    : selectedGroup?.runs[0]?.plugin.id;
  const [selectedRecord, setSelectedRecord] = useState(null);

  if (groups.length === 0) {
    return <EmptyPluginComparison />;
  }

  return (
    <Box data-testid="plugin-comparison" sx={{ display: 'grid', gap: 1.75 }}>
      {baselineOnlyGroup && (
        <PluginComparisonNotice />
      )}
      <SectionCard
        title="Plugin Comparison"
        subtitle="Runtime overhead and test health for controlled Vanilla and sanitizer executions"
      >
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1.5fr 1fr' }, gap: 1.25 }}>
          <TextField
            select
            size="small"
            label="Comparison experiment"
            value={selectedGroup.comparisonId}
            onChange={(event) => setSelectedGroupId(event.target.value)}
          >
            {groups.slice().reverse().map((group) => (
              <MenuItem key={group.comparisonId} value={group.comparisonId}>
                {shortSha(group.referenceRun)} · {formatFullDate(group.referenceRun.timestamp)} · {comparisonPluginNames(group)}
              </MenuItem>
            ))}
          </TextField>
          <TextField
            select
            size="small"
            label="Baseline plugin"
            value={baselinePluginId}
            onChange={(event) => setRequestedBaseline(event.target.value)}
          >
            {selectedGroup.runs.map((run) => (
              <MenuItem key={run.plugin.id} value={run.plugin.id}>{run.plugin.name}</MenuItem>
            ))}
          </TextField>
        </Box>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 1.2 }}>
          Same source commit, catalog, trigger, machine, environment, and target set.
        </Typography>
      </SectionCard>

      {targets.length === 0 ? (
        <Alert severity="info">Select at least one target in the global Targets filter.</Alert>
      ) : targets.map((target) => (
          <TargetComparison
            key={target}
            group={selectedGroup}
            target={target}
            suites={filters.suites}
            baselinePluginId={baselinePluginId}
            onOpen={setSelectedRecord}
          />
        ))}

      <BenchmarkResultDialog
        record={selectedRecord}
        repository={data.repository}
        onClose={() => setSelectedRecord(null)}
      />
    </Box>
  );
}
