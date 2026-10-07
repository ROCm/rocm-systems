import { Fragment } from 'react';
import { Box, Button, Checkbox, FormControl, InputLabel, MenuItem, Select, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TableSortLabel, TextField, Typography } from '@mui/material';
import { formatDuration, formatPercent } from '../../utils/formatters';
import { monoFont } from '../../theme/tokens';
import CategoryTag from '../shared/CategoryTag';
import { groupComparisonRows, rowChanges } from './branchPresentation';

function MeasuredChange({ row, measurement = 'seconds' }) {
  const { seconds, percent } = rowChanges(row);
  const direction = seconds < 0 ? 'less time' : seconds > 0 ? 'more time' : 'same time';
  return <Box sx={{ fontFamily: monoFont, fontVariantNumeric: 'tabular-nums', color: percent < -3 ? 'success.main' : percent > 3 ? 'error.main' : 'text.secondary' }}>
    {measurement === 'percent' ? (percent !== null ? formatPercent(percent) : <Typography variant="caption">Percentage unavailable: zero reference runtime</Typography>)
      : seconds === null ? '—' : <>{seconds > 0 ? '+' : seconds < 0 ? '−' : ''}{formatDuration(Math.abs(seconds))} · {direction}</>}
  </Box>;
}

function ExcludedResults({ rows, candidate, reference, target, mode }) {
  const status = (test, run) => !run ? 'No run selected' : !run.configurations.some((configuration) => configuration.target === target && configuration.mode === mode) ? 'Configuration not published'
    : !test ? 'Not in this catalog' : test.status === 'completed' ? `Completed · ${formatDuration(test.durationSeconds)}` : test.status === 'timeout' ? 'Timeout' : 'Failed';
  return <Box component="details" data-testid="branch-exclusions" sx={{ mt: 2, borderTop: 1, borderColor: 'divider' }}>
    <Box component="summary" sx={{ cursor: 'pointer', minHeight: 44, display: 'list-item', py: 1.5, ml: 2 }}>Excluded results ({rows.length})</Box>
    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>Missing catalog entries, unpublished configurations, failures and timeouts are excluded, never replaced with zero.</Typography>
    {rows.length ? rows.map((row) => {
      const test = row.candidateTest ?? row.baselineTest;
      return <Box key={test.testId} sx={{ py: 1.5, borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>{test.name}</Typography>
        <Typography variant="caption" color="text.secondary">{test.suite} · {test.logicalTestId}</Typography>
        <Typography variant="body2">Reference: {status(row.baselineTest, reference)} · Candidate: {status(row.candidateTest, candidate)}</Typography>
        {[['Reference', row.baselineTest], ['Candidate', row.candidateTest]].map(([label, result]) => result?.error && <Typography key={label} variant="caption" component="p" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', mt: 0.5 }}>{label} diagnostic: {result.error}</Typography>)}
      </Box>;
    }) : <Typography variant="body2">No excluded results in this scope.{!candidate || !reference ? ' Select both published attempts to inspect a comparison.' : ''}</Typography>}
  </Box>;
}

export default function BenchmarkDifferences({ comparison, summary, candidate, reference, selection, suites, suiteOptions, query, sort, onSuites, onQuery, onSort, onClear, onOpenComparison }) {
  const sortKey = sort?.key ?? sort ?? 'seconds';
  const sortDirection = sort?.direction ?? 'desc';
  const groups = groupComparisonRows(comparison.comparable, sortKey, sortDirection);
  const sortLabel = (key, mobile = false) => <TableSortLabel active={sortKey === key} direction={sortKey === key ? sortDirection : 'desc'}
    data-testid={`branch-${mobile ? 'mobile-' : ''}sort-${key}`} aria-label={`${key === 'seconds' ? 'Abs change' : 'Pct change'}${mobile ? ', sort benchmarks' : ''}`}
    onClick={() => onSort({ key, direction: sortKey === key && sortDirection === 'desc' ? 'asc' : 'desc' })}
    sx={{ minHeight: 44, '&.Mui-focusVisible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: 2 } }}>
    {key === 'seconds' ? 'Abs change' : 'Pct change'}
  </TableSortLabel>;
  return <Box>
    <Typography role="status" variant="body2" color="text.secondary" sx={{ mt: 1 }}>
      {summary.matched} matched · {summary.excluded} excluded
      {summary.available && <> · Base {formatDuration(comparison.baselineDuration)} → Candidate {formatDuration(comparison.candidateDuration)}</>}
    </Typography>
    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', my: 1.5, alignItems: 'center' }}>
      <FormControl size="small" sx={{ minWidth: 160, maxWidth: '100%', flex: '0 1 200px' }}>
        <InputLabel id="branch-suite-filter-label" shrink>Suites</InputLabel>
        <Select multiple displayEmpty label="Suites" labelId="branch-suite-filter-label" value={suites} onChange={(event) => onSuites(event.target.value)}
          renderValue={(chosen) => !chosen.length ? 'No suites' : chosen.length === suiteOptions.length ? 'All suites' : chosen.join(', ')}
          sx={{ minHeight: 44 }} MenuProps={{ slotProps: { paper: { sx: { maxHeight: 360, maxWidth: 'calc(100vw - 24px)' } } } }}>
          {suiteOptions.map((suite) => <MenuItem key={suite} value={suite} sx={{ minHeight: 44, overflowWrap: 'anywhere', whiteSpace: 'normal' }}><Checkbox checked={suites.includes(suite)} tabIndex={-1} size="small" />{suite}</MenuItem>)}
        </Select>
      </FormControl>
      <TextField size="small" type="search" label="Search benchmarks" value={query} onChange={(event) => onQuery(event.target.value)} sx={{ flex: '1 1 200px', '& .MuiInputBase-root': { minHeight: 44 } }} />
      {(query || suites.length !== suiteOptions.length) && <Button onClick={onClear} sx={{ minHeight: 44 }}>Reset benchmark filters</Button>}
    </Box>
    {!summary.available && <Typography variant="body2" sx={{ py: 2 }}>{summary.reason}. Choose another reference or configuration, or adjust suites and search.</Typography>}
    {summary.available && <>
      <TableContainer role="region" tabIndex={0} aria-label="Branch benchmark differences, horizontally scrollable on tablet" sx={{ minWidth: 0, maxWidth: '100%', '@media (max-width:767px)': { display: 'none' } }}>
        <Table size="small" aria-label="Benchmark differences" sx={{ minWidth: 520, tableLayout: 'fixed', '& td, & th': { px: 1, py: 1.5, verticalAlign: 'top', overflowWrap: 'anywhere' } }}>
          <TableHead><TableRow><TableCell sx={{ width: '30%' }}>Benchmark</TableCell><TableCell align="right">Base</TableCell><TableCell align="right">Candidate</TableCell>
            <TableCell align="right" sortDirection={sortKey === 'seconds' ? sortDirection : false} sx={{ width: '23%' }}>{sortLabel('seconds')}</TableCell>
            <TableCell align="right" sortDirection={sortKey === 'percent' ? sortDirection : false} sx={{ width: '20%' }}>{sortLabel('percent')}</TableCell>
          </TableRow></TableHead>
          <TableBody>{groups.map(({ suite, rows }) => <Fragment key={`${suite}:${rows[0].candidateTest.testId}`}>
            <TableRow><TableCell component="th" scope="rowgroup" colSpan={5} sx={{ bgcolor: 'action.hover' }}><CategoryTag label={suite} kind="suite" /></TableCell></TableRow>
            {rows.map((row) => <TableRow key={row.candidateTest.testId} data-testid={`branch-result-${row.candidateTest.testId}`} data-abs-change={rowChanges(row).seconds ?? "unavailable"} data-pct-change={rowChanges(row).percent ?? "unavailable"}>
              <TableCell component="th" scope="row"><Typography variant="body2">{row.candidateTest.name}</Typography><Typography variant="caption" color="text.secondary">{row.candidateTest.logicalTestId}</Typography></TableCell>
              <TableCell align="right" sx={{ fontFamily: monoFont, fontVariantNumeric: 'tabular-nums' }}>{formatDuration(row.baselineTest.durationSeconds)}</TableCell>
              <TableCell align="right" sx={{ fontFamily: monoFont, fontVariantNumeric: 'tabular-nums' }}>{formatDuration(row.candidateTest.durationSeconds)}</TableCell>
              <TableCell align="right"><MeasuredChange row={row} /></TableCell>
              <TableCell align="right"><MeasuredChange row={row} measurement="percent" /></TableCell>
            </TableRow>)}
          </Fragment>)}</TableBody>
        </Table>
      </TableContainer>
      <Box data-testid="branch-mobile-results" sx={{ display: 'none', '@media (max-width:767px)': { display: 'block' } }}>
        <Box role="group" aria-label="Sort benchmark differences" sx={{ display: 'flex', gap: 2 }}>{sortLabel('seconds', true)}{sortLabel('percent', true)}</Box>
        {groups.map(({ suite, rows }) => <Box key={`${suite}:${rows[0].candidateTest.testId}`} sx={{ mt: 1.5 }}>
          <CategoryTag label={suite} kind="suite" />
          {rows.map((row) => <Box component="article" key={row.candidateTest.testId} data-testid={`branch-mobile-result-${row.candidateTest.testId}`} sx={{ py: 1.5, borderBottom: 1, borderColor: 'divider' }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>{row.candidateTest.name}</Typography>
            <Typography variant="caption" component="div" color="text.secondary">{row.candidateTest.logicalTestId}</Typography>
            <Typography variant="body2" sx={{ fontFamily: monoFont, mt: 1 }}>Base {formatDuration(row.baselineTest.durationSeconds)} → Candidate {formatDuration(row.candidateTest.durationSeconds)}</Typography>
            <Box sx={{ mt: 0.5 }}>Abs change: <MeasuredChange row={row} />Pct change: <MeasuredChange row={row} measurement="percent" /></Box>
          </Box>)}
        </Box>)}
      </Box>
    </>}
    <Typography variant="caption" color="text.secondary" component="p" sx={{ mt: 1.5 }}>Negative change means less measured wall time. Totals include only matched completed benchmarks; environment equivalence is not guaranteed.</Typography>
    <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
      <Button variant="contained" data-testid="branch-open-comparison" disabled={!candidate || !reference || typeof onOpenComparison !== 'function'}
        onClick={() => candidate && reference && onOpenComparison({ candidateId: selection.candidateId, baselineId: selection.referenceId, target: selection.target, mode: selection.mode, suites })}
        sx={{ minHeight: 44, '@media (max-width:767px)': { width: '100%' } }}>Open full comparison ↗</Button>
    </Box>
    {(!candidate || !reference) && <Typography variant="caption" component="p" color="text.secondary">Choose both published attempts to open full comparison.</Typography>}
    <ExcludedResults rows={comparison.notComparable} candidate={candidate} reference={reference} target={selection.target} mode={selection.mode} />
  </Box>;
}
