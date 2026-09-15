import { useMemo, useState } from 'react';
import {
  Box,
  Button,
  Chip,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import ArrowDownwardRoundedIcon from '@mui/icons-material/ArrowDownwardRounded';
import ArrowUpwardRoundedIcon from '@mui/icons-material/ArrowUpwardRounded';
import RemoveRoundedIcon from '@mui/icons-material/RemoveRounded';
import InsightsRoundedIcon from '@mui/icons-material/InsightsRounded';
import SectionCard from '../shared/SectionCard';
import { selectRecentRuns } from '../../data/selectors';
import { commitTimestampFor } from '../../data/runOrdering';
import { formatDuration, formatFullDate, formatPercent, shortSha } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';
import { hasDisplayValue } from '../../utils/values';
import CommitComparison from '../shared/CommitComparison';

const runLimits = [5, 8, 12, 20];
const timeCellSx = { whiteSpace: 'nowrap', verticalAlign: 'middle' };
function TableCellContent({ children, footer, align = 'flex-start', expanded = false }) {
  return (
    <Box sx={{ minWidth: 0, minHeight: expanded ? 80 : 36, display: 'flex', alignItems: 'center' }}>
      <Box
        data-table-cell-group="true"
        sx={{ width: '100%', minWidth: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: footer ? 1 : 0 }}
      >
        <Box data-table-cell-primary="true" sx={{ minWidth: 0, minHeight: 24, display: 'flex', alignItems: 'center', justifyContent: align }}>{children}</Box>
        {footer && (
          <Box sx={{ minWidth: 0, minHeight: 24, display: 'flex', alignItems: 'center', justifyContent: align }}>
            {footer}
          </Box>
        )}
      </Box>
    </Box>
  );
}

function triggerType(value = '') {
  return value === 'manual' ? 'Manual' : 'Auto';
}

function compactRunTime(timestamp, includeZone = false) {
  const date = new Date(timestamp);
  const day = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' });
  const time = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC' });
  return `${day} · ${time}${includeZone ? ' UTC' : ''}`;
}

function ChangeValue({ value, candidate, baseline }) {
  const state = classifyDurationChange(value);
  const tone = changeTone(state);
  const color = tone === 'neutral' ? 'text.secondary' : `${tone}.main`;
  return (
    <Box>
      <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'flex-end', color, whiteSpace: 'nowrap' }}>
        {state === 'slower' && <ArrowUpwardRoundedIcon sx={{ fontSize: 15 }} />}
        {state === 'faster' && <ArrowDownwardRoundedIcon sx={{ fontSize: 15 }} />}
        {state === 'neutral' && <RemoveRoundedIcon sx={{ fontSize: 15 }} />}
        <Typography variant="body2" fontWeight={750}>{Number.isFinite(value) ? formatPercent(Math.abs(value), false) : '—'}</Typography>
      </Stack>
      {Number.isFinite(value) && <CommitComparison candidate={candidate} baseline={baseline} align="right" sx={{ mt: 0.1, fontSize: 10 }} />}
    </Box>
  );
}

function baselineDescription(row) {
  return row.baseline
    ? `Latest run with complete results for current filters from nearest earlier commit: ${shortSha(row.baseline)}`
    : 'No earlier commit has complete results for the current filters';
}

function Coverage({ row }) {
  const complete = row.total > 0 && row.completed === row.total;
  const issueCount = row.total - row.completed;
  return (
    <Stack direction="row" sx={{ alignItems: 'center', gap: 0.75, whiteSpace: 'nowrap' }}>
      <Typography variant="body2" fontWeight={700}>{row.completed}/{row.total}</Typography>
      <Chip
        size="small"
        color={complete ? 'success' : 'error'}
        variant="outlined"
        label={complete ? 'Complete' : `${issueCount} issue${issueCount === 1 ? '' : 's'}`}
      />
    </Stack>
  );
}

function CommitFlags({ row }) {
  if (row.latestCommit) {
    return (
      <Stack
        direction="row"
        role="status"
        aria-label="Latest commit"
        sx={{ minHeight: 24, alignItems: 'center', gap: 0.55, color: 'text.secondary' }}
      >
        <Box aria-hidden="true" sx={{ width: 6, height: 6, flex: '0 0 6px', borderRadius: '50%', bgcolor: 'secondary.main' }} />
        <Typography variant="caption" color="inherit" fontWeight={700}>Latest commit</Typography>
      </Stack>
    );
  }
  if (!row.olderCommit) return null;
  return (
    <Stack
      direction="row"
      role="status"
      aria-label="Historical rerun"
      sx={{ minHeight: 24, alignItems: 'center', flexWrap: 'nowrap', gap: 0.55, color: 'text.secondary' }}
    >
      <Box aria-hidden="true" sx={{ width: 6, height: 6, flex: '0 0 6px', borderRadius: '50%', bgcolor: 'warning.main' }} />
      <Typography variant="caption" color="inherit" fontWeight={700}>Historical rerun</Typography>
    </Stack>
  );
}

function RunTypeValue({ run }) {
  const type = triggerType(run.trigger);
  return (
    <Typography variant="body2" fontWeight={700} color={type === 'Manual' ? 'primary.main' : 'secondary.main'}>
      {type}
    </Typography>
  );
}

function MostRecentMarker() {
  return (
    <Stack
      component="span"
      direction="row"
      role="status"
      aria-label="Most recent run"
      sx={{
        alignItems: 'center',
        gap: 0.55,
        width: 'fit-content',
        minHeight: 24,
        color: 'text.secondary',
      }}
    >
      <Box
        component="span"
        aria-hidden="true"
        sx={{
          width: 6,
          height: 6,
          flex: '0 0 6px',
          borderRadius: '50%',
          bgcolor: 'primary.main',
        }}
      />
      <Typography component="span" variant="caption" color="inherit" fontWeight={700}>
        Most recent
      </Typography>
    </Stack>
  );
}

function CompareButton({ row, selected, onSelectRun, compact = false }) {
  const sha = shortSha(row.run);
  const label = selected ? 'Selected' : 'Compare';
  const ariaLabel = `Compare ${sha}`;
  return (
    <Button
      size="small"
      variant="outlined"
      aria-label={ariaLabel}
      aria-pressed={selected}
      onClick={() => onSelectRun(row.run.runId)}
      sx={{
        flexShrink: 0,
        minWidth: compact ? 68 : undefined,
        px: compact ? 0.75 : undefined,
        borderWidth: selected ? 2 : 1,
        fontWeight: selected ? 760 : undefined,
        '&:hover': { borderWidth: selected ? 2 : 1 },
        ...(selected && {
          '&&': {
            bgcolor: 'primary.main',
            borderColor: 'primary.main',
            color: 'primary.contrastText',
          },
          '&&:hover': {
            bgcolor: 'primary.dark',
            borderColor: 'primary.main',
          },
        }),
      }}
    >
      {label}
    </Button>
  );
}

function ViewInExplorerButton({ row, onExploreRun }) {
  return (
    <Button
      size="small"
      color="inherit"
      variant="text"
      startIcon={<InsightsRoundedIcon sx={{ fontSize: '14px !important' }} />}
      aria-label={`View run ${shortSha(row.run)} in Benchmark Explorer`}
      onClick={() => onExploreRun(row.run.runId)}
      sx={{
        minWidth: 0,
        px: 0.35,
        py: 0.1,
        fontSize: 10.5,
        color: 'text.secondary',
        whiteSpace: 'nowrap',
        '& .MuiButton-startIcon': { mr: 0.45, ml: 0 },
      }}
    >
      View in Explorer
    </Button>
  );
}

function MobileRun({ row, selected, onSelectRun, onExploreRun }) {
  return (
    <Box data-testid="mobile-run" sx={{ p: 1.5, border: 1, borderColor: selected || row.latest ? 'primary.main' : 'divider', borderRadius: 2.25, bgcolor: selected || row.latest ? 'action.hover' : 'transparent' }}>
      <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'flex-start', gap: 1 }}>
        <Box>
          <Stack direction="row" sx={{ alignItems: 'center', flexWrap: 'wrap', gap: 0.6 }}>
            <Typography component="time" dateTime={row.run.timestamp} title={formatFullDate(row.run.timestamp)} aria-label={formatFullDate(row.run.timestamp)} variant="body2" fontWeight={720}>
              {compactRunTime(row.run.timestamp, true)}
            </Typography>
            {row.latest && <MostRecentMarker />}
          </Stack>
          <Typography variant="caption" color="text.secondary" component="div">
            <Box component="span" sx={{ color: 'primary.main', fontFamily: 'monospace' }}>{shortSha(row.run)}</Box>
            {hasDisplayValue(row.run.provenance?.commitMessage) && ` · ${row.run.provenance.commitMessage}`}
          </Typography>
          {(row.latestCommit || row.olderCommit) && <Box sx={{ mt: 0.65 }}><CommitFlags row={row} /></Box>}
        </Box>
        <Stack sx={{ alignItems: 'flex-end', gap: 0.35 }}>
          <CompareButton row={row} selected={selected} onSelectRun={onSelectRun} />
          <ViewInExplorerButton row={row} onExploreRun={onExploreRun} />
        </Stack>
      </Stack>
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 1.25, mt: 1.25 }}>
        <Box>
          <Typography variant="caption" color="text.secondary">Commit time</Typography>
          <Typography component="time" dateTime={commitTimestampFor(row.run)} title={formatFullDate(commitTimestampFor(row.run))} variant="body2">
            {compactRunTime(commitTimestampFor(row.run), true)}
          </Typography>
        </Box>
        <Box><Typography variant="caption" color="text.secondary">Run type</Typography><RunTypeValue run={row.run} /></Box>
        <Box><Typography variant="caption" color="text.secondary">Coverage</Typography><Coverage row={row} /></Box>
        <Box sx={{ textAlign: 'right' }} title={baselineDescription(row)}><Typography variant="caption" color="text.secondary">Perf change</Typography><Typography variant="body2" fontWeight={700}>{formatDuration(row.duration)}</Typography><ChangeValue value={row.durationDelta} candidate={row.run} baseline={row.baseline} /></Box>
      </Box>
    </Box>
  );
}

export default function RecentRuns({ data, filters, onCompareRun, onExploreRun }) {
  const [runLimit, setRunLimit] = useState(8);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const rows = useMemo(() => selectRecentRuns(data, filters, runLimit), [data, filters, runLimit]);
  const selectRun = (runId) => {
    if (selectedRunId === runId) {
      setSelectedRunId(null);
      return;
    }
    if (!selectedRunId) {
      setSelectedRunId(runId);
      return;
    }
    onCompareRun([selectedRunId, runId]);
  };
  return (
    <SectionCard
      title="Recent Runs"
      subtitle={(
        <>
          <Box component="span" sx={{ display: 'block' }}>Each perf change uses the latest run from the nearest earlier commit that completed all currently selected tests.</Box>
          <Box component="span" sx={{ display: 'block' }}>To compare runs, select the candidate first and the baseline second.</Box>
        </>
      )}
      action={(
        <FormControl size="small" sx={{ width: { xs: '100%', sm: 128 }, flexShrink: 0 }}>
          <InputLabel id="run-limit-label">Show runs</InputLabel>
          <Select
            labelId="run-limit-label"
            label="Show runs"
            value={runLimit}
            SelectDisplayProps={{ 'aria-label': 'Show runs' }}
            renderValue={(value) => `${value} runs`}
            onChange={(event) => {
              setRunLimit(Number(event.target.value));
              setSelectedRunId(null);
            }}
          >
            {runLimits.map((limit) => <MenuItem key={limit} value={limit}>{limit} runs</MenuItem>)}
          </Select>
        </FormControl>
      )}
    >
      {selectedRunId && (
        <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', gap: 1.5, mb: 1.25, p: 1.15, borderRadius: 2, bgcolor: 'action.hover', border: 1, borderColor: 'primary.main' }}>
          <Box>
            <Typography variant="body2" fontWeight={720}>Candidate selected</Typography>
            <Typography variant="caption" color="text.secondary">Select another run as the baseline to open Run Comparison.</Typography>
          </Box>
          <Button size="small" color="inherit" onClick={() => setSelectedRunId(null)}>Clear</Button>
        </Stack>
      )}
      <TableContainer data-testid="recent-runs-table" sx={{ display: { xs: 'none', md: 'block' }, borderTop: 1, borderColor: 'divider', overflowX: 'auto' }}>
        <Table size="small" sx={{ width: '100%', minWidth: 1100, tableLayout: 'fixed', '& .MuiTableCell-root': { px: { md: 1, lg: 1.5 } } }}>
          <colgroup>
            <col style={{ width: '10%' }} /><col style={{ width: '11%' }} /><col style={{ width: '33%' }} /><col style={{ width: '11%' }} /><col style={{ width: '7%' }} /><col style={{ width: '9%' }} /><col style={{ width: '7%' }} /><col style={{ width: '12%' }} />
          </colgroup>
          <TableHead>
            <TableRow>
              <TableCell>Actions</TableCell><TableCell>Run time (UTC)</TableCell><TableCell>Commit</TableCell><TableCell>Commit time (UTC)</TableCell><TableCell>Run type</TableCell><TableCell>Coverage</TableCell><TableCell align="right">Duration</TableCell><TableCell align="right" title="Latest earlier commit with complete results for the current filters">Perf change</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((row) => {
              const selected = selectedRunId === row.run.runId;
              const expanded = row.latest || row.latestCommit || row.olderCommit;
              return (
                <TableRow key={row.run.runId} hover selected={selected} sx={{ bgcolor: row.latest ? 'action.hover' : 'transparent' }}>
                  <TableCell sx={{ verticalAlign: 'middle' }}>
                    <Stack sx={{ alignItems: 'flex-start', gap: 0.25 }}>
                      <CompareButton row={row} selected={selected} onSelectRun={selectRun} compact />
                      <ViewInExplorerButton row={row} onExploreRun={onExploreRun} />
                    </Stack>
                  </TableCell>
                  <TableCell sx={timeCellSx}>
                    <TableCellContent expanded={expanded} footer={row.latest ? <MostRecentMarker /> : null}>
                      <Typography component="time" dateTime={row.run.timestamp} title={formatFullDate(row.run.timestamp)} aria-label={formatFullDate(row.run.timestamp)} variant="body2">
                        {compactRunTime(row.run.timestamp)}
                      </Typography>
                    </TableCellContent>
                  </TableCell>
                  <TableCell sx={{ minWidth: 0, verticalAlign: 'middle' }}>
                    <TableCellContent expanded={expanded} footer={row.latestCommit || row.olderCommit ? <CommitFlags row={row} /> : null}>
                      <Stack direction="row" sx={{ minWidth: 0, alignItems: 'baseline', gap: 1 }}>
                        <Typography component="code" variant="body2" color="primary.main" fontWeight={760} sx={{ flexShrink: 0 }}>{shortSha(row.run)}</Typography>
                        {hasDisplayValue(row.run.provenance?.commitMessage) && <Typography variant="body2" color="text.secondary" noWrap sx={{ minWidth: 0 }}>{row.run.provenance.commitMessage}</Typography>}
                      </Stack>
                    </TableCellContent>
                  </TableCell>
                  <TableCell sx={timeCellSx}>
                    <TableCellContent expanded={expanded}>
                      <Typography component="time" dateTime={commitTimestampFor(row.run)} title={formatFullDate(commitTimestampFor(row.run))} variant="body2">
                        {compactRunTime(commitTimestampFor(row.run))}
                      </Typography>
                    </TableCellContent>
                  </TableCell>
                  <TableCell sx={{ verticalAlign: 'middle' }}><TableCellContent expanded={expanded}><RunTypeValue run={row.run} /></TableCellContent></TableCell>
                  <TableCell sx={{ verticalAlign: 'middle' }}><TableCellContent expanded={expanded}><Coverage row={row} /></TableCellContent></TableCell>
                  <TableCell align="right" sx={{ verticalAlign: 'middle' }}>
                    <TableCellContent expanded={expanded} align="flex-end"><Typography variant="body2" fontWeight={700}>{formatDuration(row.duration)}</Typography></TableCellContent>
                  </TableCell>
                  <TableCell align="right" title={baselineDescription(row)} sx={{ verticalAlign: 'middle' }}><TableCellContent expanded={expanded} align="flex-end"><ChangeValue value={row.durationDelta} candidate={row.run} baseline={row.baseline} /></TableCellContent></TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
      <Stack sx={{ display: { xs: 'flex', md: 'none' }, gap: 1 }}>
        {rows.map((row) => (
          <MobileRun
            key={row.run.runId}
            row={row}
            selected={selectedRunId === row.run.runId}
            onSelectRun={selectRun}
            onExploreRun={onExploreRun}
          />
        ))}
      </Stack>
    </SectionCard>
  );
}
