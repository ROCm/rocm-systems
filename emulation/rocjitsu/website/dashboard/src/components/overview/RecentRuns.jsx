import { useMemo, useState } from 'react';
import { Box, Link, Pagination, PaginationItem, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material';
import SectionCard from '../shared/SectionCard';
import { monoFont } from '../../theme/tokens';
import { selectRecentRunAttempts, selectRecentRunSummaries } from '../../data/selectors';
import { commitTimestampFor } from '../../data/runOrdering';
import { formatDuration, formatFullDate, shortSha } from '../../utils/formatters';
import { recentRunsPage } from './overviewPresentation';


function sourceCommitUrl(run, publicationRepository) {
  const sha = run.source?.commit ?? run.provenance?.rocjitsuCommitSha;
  // Legacy records lack source metadata; only those inherit the publication repository.
  const repository = run.source?.repository ?? publicationRepository;
  if (typeof sha !== 'string' || !/^[a-fA-F0-9]{7,40}$/.test(sha) || typeof repository !== 'string') return null;
  try {
    const url = new URL(repository);
    if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password) return null;
    url.pathname = `${url.pathname.replace(/\/$/, '')}/commit/${sha}`;
    url.search = '';
    url.hash = '';
    return url.href;
  } catch {
    return null;
  }
}

function RunTime({ timestamp }) {
  if (!timestamp) return '—';
  return <Typography component="time" dateTime={timestamp} title={formatFullDate(timestamp)} aria-label={formatFullDate(timestamp)} variant="caption" sx={{ whiteSpace: 'nowrap', fontSize: 12 }}>{new Date(timestamp).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC' })}</Typography>;
}

function statusFor(row) {
  if (row.failed > 0 || row.timeout > 0) return { label: '× Fail', tone: 'error' };
  if (row.total > 0 && row.completed === row.total) return { label: '✓ OK', tone: 'success' };
  return { label: 'Unavailable', tone: 'neutral' };
}

export default function RecentRuns({ data, filters }) {
  const attempts = useMemo(() => selectRecentRunAttempts(data), [data]);
  const [requestedPage, setPage] = useState(0);
  const { page, start, end } = recentRunsPage(attempts.length, requestedPage);
  // Clamp before rendering; retain the corrected page if history grows again.
  if (requestedPage !== page) setPage(page);
  const rows = useMemo(() => selectRecentRunSummaries(attempts.slice(start, end), filters), [attempts, start, end, filters]);
  return (
    <SectionCard title="Recent Runs" subtitle="Published develop runs · newest executions first · 20 per page · selected targets, suites and execution modes" data-testid="recent-runs" sx={{ minWidth: 0 }} contentSx={{ p: 2, '&:last-child': { pb: 2 }, minWidth: 0 }}>
      {rows.length > 0 ? <>
        <TableContainer key={page} data-testid="recent-runs-table" tabIndex={0} role="region" aria-label="Recent runs scroll area" sx={{ height: rows.length > 5 ? 344 : 'auto', maxHeight: 344, maxWidth: '100%', overflow: 'auto', scrollbarGutter: 'stable', overscrollBehavior: 'contain', '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: 2 } }}>
          <Table stickyHeader size="small" aria-label="Recent runs" sx={{ tableLayout: 'fixed', minWidth: 760, borderCollapse: 'separate', borderSpacing: 0, '& td, & th': { boxSizing: 'border-box', px: 1.5 }, '& th': { height: 44, py: 1, whiteSpace: 'nowrap', fontSize: 11, bgcolor: 'action.hover' }, '& td': { height: 60, py: 1, fontSize: 12, whiteSpace: 'nowrap' }, '& td:not(:first-of-type), & th:not(:first-of-type)': { textAlign: 'center' } }}>
            <colgroup>{Array.from({ length: 6 }, (_, i) => <col key={i} style={{ width: '16.6667%' }} />)}</colgroup>
            <TableHead><TableRow>{['Commit', 'Run time (UTC)', 'Commit time (UTC)', 'Coverage', 'Duration', 'Status'].map((label) => <TableCell key={label}>{label}</TableCell>)}</TableRow></TableHead>
            <TableBody>{rows.map((row) => {
              const status = statusFor(row);
              const full = row.total > 0 && row.completed === row.total;
              const commitSha = row.run.source?.commit ?? row.run.provenance?.rocjitsuCommitSha;
              const commitUrl = sourceCommitUrl(row.run, data.repository);
              return <TableRow key={row.run.runId} hover data-run-id={row.run.runId}>
                <TableCell>
                  {commitUrl ? <Link href={commitUrl} target="_blank" rel="noopener noreferrer" color="primary" underline="hover" title={commitSha} aria-label={`Open source commit ${commitSha} (opens in new tab)`} sx={{ display: 'block', fontSize: 12, fontFamily: monoFont, lineHeight: '18px' }}>{commitSha.slice(0, 8)}</Link> : <Typography component="code" variant="body2" title={row.run.runId} sx={{ display: 'block', fontSize: 12, fontFamily: monoFont, color: 'text.secondary', lineHeight: '18px' }}>{shortSha(row.run)}</Typography>}
                  <Typography variant="caption" component="span" title={row.run.provenance?.commitMessage} sx={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 11, lineHeight: '18px', color: 'text.secondary' }}>{row.run.provenance?.commitMessage ?? 'No commit message provided'}</Typography>
                </TableCell>
                <TableCell><RunTime timestamp={row.run.timestamp} /></TableCell>
                <TableCell><RunTime timestamp={commitTimestampFor(row.run)} /></TableCell>
                <TableCell sx={{ fontFamily: monoFont, color: full ? 'success.main' : 'text.secondary' }}>{row.total ? `${row.completed}/${row.total}` : <Box component="span" title="No selected tests">—</Box>}</TableCell>
                <TableCell sx={{ fontFamily: monoFont }}>{formatDuration(row.duration)}</TableCell>
                <TableCell><Box component="span" data-tone={status.tone} sx={{ fontWeight: 600, color: status.tone === 'neutral' ? 'text.secondary' : `${status.tone}.main` }}>{status.label}</Box></TableCell>
              </TableRow>;
            })}</TableBody>
          </Table>
        </TableContainer>
        <Stack direction="row" sx={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mt: 1.5 }}>
          <Typography role="status" aria-live="polite" variant="caption" color="text.secondary">Showing {start + 1}–{start + rows.length} of {attempts.length} runs</Typography>
          <Pagination
            aria-label="Recent runs pages"
            count={Math.ceil(attempts.length / 20)} page={page + 1}
            onChange={(_event, value) => setPage(value - 1)}
            showFirstButton showLastButton siblingCount={0} boundaryCount={1} size="small"
            renderItem={(item) => <PaginationItem {...item} title={['first', 'previous', 'next', 'last'].includes(item.type) ? `Go to ${item.type} page` : undefined} />}
            sx={{ '& .MuiPagination-ul': { flexWrap: 'wrap' } }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ width: '100%' }}>Scroll within this page · Tables scroll locally on phones</Typography>
        </Stack>
      </> : <Typography color="text.secondary" sx={{ py: 3 }}>No runs available.</Typography>}
    </SectionCard>
  );
}
