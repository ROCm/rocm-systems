import { useMemo, useState } from 'react';
import {
  Box,
  ButtonBase,
  InputAdornment,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
  TableSortLabel,
  TextField,
  Typography,
} from '@mui/material';
import SearchRoundedIcon from '@mui/icons-material/SearchRounded';
import BenchmarkResultDialog from '../benchmarks/BenchmarkResultDialog';
import StatusChip from '../shared/StatusChip';
import { formatDuration, formatPercent, formatProblem } from '../../utils/formatters';
import { detailActionStyles } from '../../theme/styles';
import CommitComparison from '../shared/CommitComparison';

const hiddenBelowLaptop = { display: { xs: 'none', lg: 'table-cell' } };
const hiddenBelowTablet = { display: { xs: 'none', md: 'table-cell' } };
const rightAlignedColumn = { pr: 4 };
const rowsPerPageOptions = [10, 25, 50];
const resultColumns = [
  { key: 'target', label: 'Target' },
  { key: 'suite', label: 'Suite' },
  { key: 'benchmark', label: 'Benchmark' },
  { key: 'type', label: 'Type', sx: hiddenBelowLaptop },
  { key: 'problem', label: 'Problem', sx: hiddenBelowLaptop },
  { key: 'duration', label: 'Duration', align: 'right', sx: rightAlignedColumn },
  { key: 'baseline', label: 'Baseline', align: 'right', sx: { ...hiddenBelowTablet, ...rightAlignedColumn } },
  { key: 'change', label: 'Change', align: 'right', sx: rightAlignedColumn },
  { key: 'status', label: 'Status' },
];
const resultCollator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

function resultSortValue(row, key) {
  if (key === 'benchmark') return row.name;
  if (key === 'type') return row.problem?.dataType;
  if (key === 'problem') return formatProblem(row.problem);
  if (key === 'duration') return row.durationSeconds;
  if (key === 'baseline') return row.previous?.durationSeconds;
  if (key === 'change') return row.delta;
  return row[key];
}

function compareResultRows(left, right, key, direction) {
  const leftValue = resultSortValue(left, key);
  const rightValue = resultSortValue(right, key);
  const leftMissing = leftValue == null || leftValue === '' || (typeof leftValue === 'number' && !Number.isFinite(leftValue));
  const rightMissing = rightValue == null || rightValue === '' || (typeof rightValue === 'number' && !Number.isFinite(rightValue));

  if (leftMissing || rightMissing) {
    if (leftMissing && rightMissing) return 0;
    return leftMissing ? 1 : -1;
  }

  const comparison = typeof leftValue === 'number' && typeof rightValue === 'number'
    ? leftValue - rightValue
    : resultCollator.compare(String(leftValue), String(rightValue));
  return direction === 'asc' ? comparison : -comparison;
}

function SortableHeader({ column, sortKey, sortDirection, onSort }) {
  const active = sortKey === column.key;
  return (
    <TableCell
      align={column.align}
      sx={column.sx}
      sortDirection={active ? sortDirection : false}
    >
      <TableSortLabel
        active={active}
        direction={active ? sortDirection : 'asc'}
        onClick={() => onSort(column.key)}
        sx={{
          flexDirection: 'row',
          ...(column.align === 'right' && {
            position: 'relative',
            '& .MuiTableSortLabel-icon': {
              position: 'absolute',
              right: -22,
              m: 0,
            },
          }),
        }}
      >
        {column.label}
      </TableSortLabel>
    </TableCell>
  );
}

export default function ResultsTable({ results, run, baseline, repository, search, onSearch }) {
  const [selected, setSelected] = useState(null);
  const [sortKey, setSortKey] = useState(null);
  const [sortDirection, setSortDirection] = useState('asc');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(rowsPerPageOptions[0]);
  const filteredRows = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return results;
    return results.filter((row) => [row.target, row.suite, row.name, row.problem?.operation, row.problem?.dataType].some((value) => String(value).toLowerCase().includes(query)));
  }, [results, search]);
  const sortedRows = useMemo(() => {
    if (!sortKey) return filteredRows;
    return filteredRows
      .map((row, index) => ({ row, index }))
      .sort((left, right) => (
        compareResultRows(left.row, right.row, sortKey, sortDirection)
        || left.index - right.index
      ))
      .map((item) => item.row);
  }, [filteredRows, sortDirection, sortKey]);
  const lastPage = Math.max(0, Math.ceil(sortedRows.length / rowsPerPage) - 1);
  const visiblePage = Math.min(page, lastPage);
  const rows = useMemo(() => {
    const start = visiblePage * rowsPerPage;
    return sortedRows.slice(start, start + rowsPerPage);
  }, [rowsPerPage, sortedRows, visiblePage]);

  const sortBy = (key) => {
    setPage(0);
    if (sortKey === key) {
      setSortDirection((current) => (current === 'asc' ? 'desc' : 'asc'));
      return;
    }
    setSortKey(key);
    setSortDirection('asc');
  };

  return (
    <Paper data-testid="latest-results" variant="outlined" sx={{ overflow: 'hidden', borderRadius: 3, boxShadow: 1 }}>
      <Stack direction={{ xs: 'column', sm: 'row' }} sx={{ justifyContent: 'space-between', alignItems: { xs: 'stretch', sm: 'center' }, gap: 1.5, p: { xs: 2, sm: 2.5 } }}>
        <Box>
          <Typography variant="h2">Latest Results</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.4 }}>
            Results from the Overview run · Select a benchmark name for details
          </Typography>
        </Box>
        <TextField
          size="small"
          value={search}
          onChange={(event) => {
            setPage(0);
            onSearch(event.target.value);
          }}
          placeholder="Search benchmarks"
          slotProps={{
            htmlInput: { 'aria-label': 'Search benchmark results' },
            input: { startAdornment: <InputAdornment position="start"><SearchRoundedIcon fontSize="small" /></InputAdornment> },
          }}
          sx={{ width: { xs: '100%', sm: 290 } }}
        />
      </Stack>
      <TableContainer sx={{ maxHeight: 510, borderTop: 1, borderColor: 'divider' }}>
        <Table stickyHeader size="small" sx={{ minWidth: 880 }}>
          <TableHead>
            <TableRow>
              {resultColumns.map((column) => (
                <SortableHeader
                  key={column.key}
                  column={column}
                  sortKey={sortKey}
                  sortDirection={sortDirection}
                  onSort={sortBy}
                />
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.testId} sx={{ '&:last-child td': { borderBottom: 0 } }}>
                <TableCell><Typography variant="body2" fontWeight={700} color="primary.main">{row.target}</Typography></TableCell>
                <TableCell>{row.suite}</TableCell>
                <TableCell>
                  <ButtonBase
                    aria-label={`Open ${row.name} result details for ${row.target}`}
                    onClick={() => setSelected(row)}
                    sx={detailActionStyles}
                  >
                    <Typography variant="body2" fontWeight={650}>{row.name}</Typography>
                  </ButtonBase>
                </TableCell>
                <TableCell sx={hiddenBelowLaptop}>{row.problem?.dataType?.toUpperCase() ?? '—'}</TableCell>
                <TableCell sx={hiddenBelowLaptop}>{formatProblem(row.problem)}</TableCell>
                <TableCell align="right" sx={{ ...rightAlignedColumn, fontVariantNumeric: 'tabular-nums', fontWeight: 650 }}>{formatDuration(row.durationSeconds)}</TableCell>
                <TableCell align="right" sx={{ ...hiddenBelowTablet, ...rightAlignedColumn, fontVariantNumeric: 'tabular-nums', color: 'text.secondary' }}>{formatDuration(row.previous?.durationSeconds)}</TableCell>
                <TableCell align="right" sx={rightAlignedColumn}>
                  <Typography variant="body2" fontWeight={720} color={!row.comparable ? 'text.secondary' : row.delta > 0 ? 'error.main' : 'success.main'}>
                    {formatPercent(row.delta)}
                  </Typography>
                  {row.comparable && <CommitComparison candidate={run} baseline={baseline} align="right" sx={{ mt: 0.15, fontSize: 10 }} />}
                </TableCell>
                <TableCell><StatusChip status={row.status} /></TableCell>
              </TableRow>
            ))}
            {rows.length === 0 && (
              <TableRow><TableCell colSpan={9} align="center" sx={{ py: 6, color: 'text.secondary' }}>No benchmarks match this search.</TableCell></TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        component="div"
        count={sortedRows.length}
        page={visiblePage}
        rowsPerPage={rowsPerPage}
        rowsPerPageOptions={rowsPerPageOptions}
        onPageChange={(_, nextPage) => setPage(nextPage)}
        onRowsPerPageChange={(event) => {
          setRowsPerPage(Number(event.target.value));
          setPage(0);
        }}
        labelDisplayedRows={({ from, to, count }) => `${from}–${to} of ${count} results`}
        showFirstButton
        showLastButton
        sx={{
          borderTop: 1,
          borderColor: 'divider',
          '& .MuiTablePagination-toolbar': { minHeight: 54, flexWrap: 'wrap', justifyContent: 'flex-end' },
          '& .MuiTablePagination-spacer': { display: { xs: 'none', sm: 'block' } },
        }}
      />

      <BenchmarkResultDialog
        record={selected ? { run, test: selected } : null}
        repository={repository}
        onClose={() => setSelected(null)}
      />
    </Paper>
  );
}
