import { useMemo, useState } from 'react';
import {
  Box,
  ButtonBase,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
  Typography,
} from '@mui/material';
import StatusChip from '../shared/StatusChip';
import { selectBenchmarkRecords } from '../../data/selectors';
import { commitTimestampFor } from '../../data/runOrdering';
import { formatDuration, formatFullDate, formatPercent, formatShortDate, shortSha } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';
import { detailActionStyles } from '../../theme/styles';
import { hasDisplayValue } from '../../utils/values';
import CommitComparison from '../shared/CommitComparison';

const rowsPerPageOptions = [25, 50, 100];

function ChangeValue({ value, candidate, baseline }) {
  const tone = changeTone(classifyDurationChange(value));
  const color = Number.isFinite(value) && tone !== 'neutral' ? `${tone}.main` : 'text.secondary';
  return (
    <Box>
      <Typography variant="body2" color={color} fontWeight={720}>{formatPercent(value)}</Typography>
      {Number.isFinite(value) && <CommitComparison candidate={candidate} baseline={baseline} align="right" sx={{ mt: 0.1, fontSize: 10 }} />}
    </Box>
  );
}

export default function HistoricalRecords({ data, filters, benchmark, onSelectRecord }) {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(rowsPerPageOptions[0]);
  const pageData = useMemo(() => selectBenchmarkRecords(
    data,
    filters,
    benchmark.id,
    page * rowsPerPage,
    rowsPerPage,
  ), [benchmark.id, data, filters, page, rowsPerPage]);
  const columnWidths = ['20%', '28%', '12%', '13%', '12%', '15%'];

  return (
    <Paper data-testid="historical-records" variant="outlined" sx={{ borderRadius: 3, overflow: 'hidden' }}>
      <Box sx={{ p: 2.5 }}>
        <Typography variant="h2">Benchmark Run History</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.4 }}>
          Each change compares with the latest completed result for the same benchmark and target from the nearest earlier commit. Select a run time for details.
        </Typography>
      </Box>
      <TableContainer sx={{ maxHeight: 470, borderTop: 1, borderColor: 'divider' }}>
        <Table size="small" stickyHeader sx={{ minWidth: 760, tableLayout: 'fixed' }}>
          <colgroup>
            {columnWidths.map((width, index) => <col key={`${index}:${width}`} style={{ width }} />)}
          </colgroup>
          <TableHead>
            <TableRow>
              <TableCell>Run time</TableCell><TableCell>Commit</TableCell><TableCell>Target</TableCell><TableCell align="right">Duration</TableCell><TableCell align="right" title="Latest completed result for the same benchmark and target from an earlier commit">Change</TableCell><TableCell>Status</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {pageData.records.map((record) => (
              <TableRow
                key={`${record.run.runId}:${record.test.testId}`}
                sx={{ '&:last-child td': { borderBottom: 0 } }}
              >
                <TableCell>
                  <ButtonBase
                    aria-label={`Open result details for ${shortSha(record.run)} at ${formatFullDate(record.run.timestamp)}`}
                    onClick={() => onSelectRecord(record)}
                    sx={detailActionStyles}
                  >
                    <Typography variant="body2">{formatFullDate(record.run.timestamp)}</Typography>
                  </ButtonBase>
                </TableCell>
                <TableCell>
                  <Stack direction="row" sx={{ minWidth: 0, alignItems: 'baseline', gap: 0.75 }}>
                    <Typography component="code" variant="caption" color="primary.main" sx={{ flexShrink: 0 }}>{shortSha(record.run)}</Typography>
                    {hasDisplayValue(record.run.provenance?.commitMessage) && (
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        noWrap
                        title={record.run.provenance.commitMessage}
                        sx={{ minWidth: 0 }}
                      >
                        {record.run.provenance.commitMessage}
                      </Typography>
                    )}
                  </Stack>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>{formatShortDate(commitTimestampFor(record.run))}</Typography>
                </TableCell>
                <TableCell>{record.test.target}</TableCell>
                <TableCell align="right" sx={{ fontVariantNumeric: 'tabular-nums', fontWeight: 650 }}>{formatDuration(record.test.durationSeconds)}</TableCell>
                <TableCell align="right" sx={{ fontVariantNumeric: 'tabular-nums' }}><ChangeValue value={record.delta} candidate={record.run} baseline={record.baseline} /></TableCell>
                <TableCell><StatusChip status={record.test.status} /></TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        component="div"
        count={pageData.total}
        page={page}
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
    </Paper>
  );
}
