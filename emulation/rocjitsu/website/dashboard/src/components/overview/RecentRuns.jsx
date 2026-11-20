import { tableStyles } from '../../theme/tableStyles';
import { fontSizes, fontWeights, textStyles, sansFont } from '../../theme/typography';
import { useMemo, useState } from 'react';
import { Box, Link, Pagination, Stack, Table, TableBody, TableCell, TableHead, TableRow, Typography } from '@mui/material';
import SectionCard from '../shared/SectionCard';
import { selectRecentRunAttempts, selectRecentRunSummaries } from '../../data/selectors';
import { commitTimestampFor } from '../../data/runOrdering';
import { formatCompactRunTime, formatDuration, formatFullDate, shortSha } from '../../utils/formatters';
import { recentRunsPage } from './overviewPresentation';
import { commitUrl } from '../../utils/commitUrl';
import { isValidTimestamp } from '../../utils/timestamp';
import ScrollTableContainer from '../shared/ScrollTableContainer';


function sourceCommitUrl(run, publicationRepository) {
  const sha = run.source?.commit ?? run.provenance?.rocjitsuCommitSha;
  // Legacy records lack source metadata; only those inherit the publication repository.
  const repository = run.source?.repository ?? publicationRepository;
  return commitUrl(repository, sha);
}

function RunTime({ timestamp }) {
  if (!isValidTimestamp(timestamp)) return '-';
  return <Typography component="time" dateTime={timestamp} title={formatFullDate(timestamp)} aria-label={formatFullDate(timestamp)} variant="body2" sx={{ whiteSpace: 'nowrap' }}>{formatCompactRunTime(timestamp)}</Typography>;
}

function statusFor(row) {
  if (row.failed > 0 || row.timeout > 0) return { label: '× Fail', tone: 'error' };
  if (row.total > 0 && row.completed === row.total) return { label: '✓ OK', tone: 'success' };
  return { label: 'Unavailable', tone: 'neutral' };
}

export default function RecentRuns({ data, filters }) {
  const attempts = useMemo(() => selectRecentRunAttempts(data), [data]);
  const [requestedPage, setPage] = useState(0);
  const { page, start, end, pageCount } = recentRunsPage(attempts.length, requestedPage);
  // Clamp before rendering; retain the corrected page if history grows again.
  if (requestedPage !== page) setPage(page);
  const rows = useMemo(() => selectRecentRunSummaries(attempts.slice(start, end), filters), [attempts, start, end, filters]);
  return (
    <SectionCard title="Recent runs" subtitle="Published develop runs · newest executions first · selected targets, suites and thread policies · local time" data-testid="recent-runs" sx={{ minWidth: 0 }} contentSx={{ minWidth: 0 }}>
      {rows.length > 0 ? <>
        <ScrollTableContainer key={page} data-testid="recent-runs-table" tabIndex={0} role="region" aria-label="Recent runs scroll area" sx={{ height: rows.length > 7 ? 496 : 'auto', maxHeight: 496, scrollbarGutter: 'stable', overscrollBehavior: 'contain' }}>
          <Table stickyHeader size="small" aria-label="Recent runs" sx={{ ...tableStyles.recentRuns, tableLayout: 'fixed', minWidth: 1040, borderCollapse: 'separate', borderSpacing: 0, '& td:not(:first-of-type), & th:not(:first-of-type)': { textAlign: 'center' } }}>
            <colgroup><col style={{ width: '35%' }} />{Array.from({ length: 5 }, (_, i) => <col key={i} style={{ width: '13%' }} />)}</colgroup>
            <TableHead><TableRow>{['Commit', 'Run time (local time)', 'Commit time (local time)', 'Coverage', 'Duration', 'Status'].map((label) => <TableCell key={label}>{label}</TableCell>)}</TableRow></TableHead>
            <TableBody>{rows.map((row) => {
              const status = statusFor(row);
              const full = row.total > 0 && row.completed === row.total;
              const commitSha = row.run.source?.commit ?? row.run.provenance?.rocjitsuCommitSha;
              const sourceUrl = sourceCommitUrl(row.run, data.repository);
              return <TableRow key={row.run.runId} hover data-run-id={row.run.runId}>
                <TableCell>
                  {sourceUrl ? <Link href={sourceUrl} target="_blank" rel="noopener noreferrer" color="primary" underline="hover" aria-label={`Open source commit ${commitSha} (opens in new tab)`} sx={{ display: 'block', fontSize: fontSizes.body, fontFamily: sansFont, fontWeight: fontWeights.medium, color: 'primary.main', lineHeight: '18px' }}>{commitSha.slice(0, 8)}</Link> : <Typography component="code" variant="body2" sx={{ display: 'block', fontSize: fontSizes.body, fontFamily: sansFont, color: 'dashboard.commitText', lineHeight: '18px' }}>{shortSha(row.run)}</Typography>}
                  <Typography variant="caption" component="span" onMouseEnter={(event) => { const el = event.currentTarget; el.title = el.scrollWidth > el.clientWidth ? row.run.provenance?.commitMessage ?? '' : ''; }} sx={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: fontSizes.caption, lineHeight: '18px', color: 'text.secondary' }}>{row.run.provenance?.commitMessage ?? 'No commit message provided'}</Typography>
                </TableCell>
                <TableCell><RunTime timestamp={row.run.timestamp} /></TableCell>
                <TableCell><RunTime timestamp={commitTimestampFor(row.run)} /></TableCell>
                <TableCell sx={{ ...textStyles.numeric, color: full ? 'success.main' : 'text.secondary' }}>{row.total ? `${row.completed}/${row.total}` : <Box component="span">—</Box>}</TableCell>
                <TableCell sx={textStyles.numeric}>{formatDuration(row.duration)}</TableCell>
                <TableCell><Box component="span" data-tone={status.tone} sx={{ fontWeight: fontWeights.emphasized, color: status.tone === 'neutral' ? 'text.secondary' : `${status.tone}.main` }}>{status.label}</Box></TableCell>
              </TableRow>;
            })}</TableBody>
          </Table>
        </ScrollTableContainer>
        <Stack direction="row" sx={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mt: '12px' }}>
          <Typography role="status" aria-live="polite" variant="caption" color="text.secondary">Showing {start + 1}–{start + rows.length} of {attempts.length} runs</Typography>
          <Pagination
            aria-label="Recent runs pages"
            count={pageCount} page={page + 1}
            onChange={(_event, value) => setPage(value - 1)}
            showFirstButton showLastButton siblingCount={0} boundaryCount={1} size="small"
            sx={{ '& .MuiPagination-ul': { flexWrap: 'wrap' } }}
          />

        </Stack>
      </> : <Typography color="text.secondary" sx={{ py: 3 }}>No runs available.</Typography>}
    </SectionCard>
  );
}
