import { useState } from 'react';
import { Alert, Box, Chip, Pagination, Paper, Stack, Typography } from '@mui/material';
import ErrorOutlineRoundedIcon from '@mui/icons-material/ErrorOutlineRounded';
import RunReliability from '../failures/RunReliability';
import StatusChip from '../shared/StatusChip';
import { selectFailures, selectRunReliability } from '../../data/selectors';
import { formatFullDate, shortSha } from '../../utils/formatters';

const CASES_PER_PAGE = 20;

export default function FailuresView({ data, filters }) {
  const failures = selectFailures(data, filters);
  const reliability = selectRunReliability(data, filters);
  const pageCount = Math.max(1, Math.ceil(failures.length / CASES_PER_PAGE));
  // A filter change produces a different case list, so paging restarts with it.
  const [page, setPage] = useState({ number: 1, filters });
  const activePage = Math.min(page.filters === filters ? page.number : 1, pageCount);
  const start = (activePage - 1) * CASES_PER_PAGE;
  const visible = failures.slice(start, start + CASES_PER_PAGE);

  return (
    <Stack sx={{ gap: 1.75 }}>
      <RunReliability reliability={reliability} />
      {failures.length === 0 ? (
        <Alert severity="success">No failed or timed-out cases match the current filters.</Alert>
      ) : (
        <Paper data-testid="failure-cases" variant="outlined" sx={{ p: { xs: 2, sm: 2.5 }, borderRadius: 3 }}>
          <Stack direction="row" sx={{ alignItems: 'center', gap: 1, mb: 2.25 }}>
            <ErrorOutlineRoundedIcon color="error" />
            <Box>
              <Typography variant="h2">Failed and Timed-Out Cases</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.35 }}>Newest first · Current filters</Typography>
            </Box>
            <Chip data-testid="failure-total" label={failures.length} color="error" size="small" sx={{ ml: 'auto' }} />
          </Stack>
          <Stack sx={{ gap: 1 }}>
            {visible.map(({ run, test }) => (
              <Paper data-testid="failure-case" key={`${run.runId}:${test.testId}`} variant="outlined" sx={{ p: 1.75, bgcolor: 'action.hover' }}>
                <Stack direction={{ xs: 'column', sm: 'row' }} sx={{ justifyContent: 'space-between', alignItems: { xs: 'flex-start', sm: 'center' }, gap: 1.5 }}>
                  <Box>
                    <Stack direction="row" sx={{ alignItems: 'center', gap: 1, mb: 0.55 }}>
                      <Typography variant="body2" fontWeight={700}>{test.name}</Typography>
                      <StatusChip status={test.status} />
                    </Stack>
                    <Typography variant="caption" color="text.secondary">{test.target} · {test.suite} · {test.error}</Typography>
                  </Box>
                  <Box data-failure-time={run.timestamp} sx={{ textAlign: { sm: 'right' }, flexShrink: 0 }}>
                    <Typography variant="caption" sx={{ display: 'block' }}>{formatFullDate(run.timestamp)}</Typography>
                    <Typography variant="caption" color="primary.main" component="code">{shortSha(run)}</Typography>
                  </Box>
                </Stack>
              </Paper>
            ))}
          </Stack>
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            sx={{ alignItems: { sm: 'center' }, justifyContent: 'space-between', gap: 1, mt: 1.75 }}
          >
            <Typography data-testid="failure-range" variant="caption" color="text.secondary">
              {start + 1}–{start + visible.length} of {failures.length} cases
            </Typography>
            {pageCount > 1 && (
              <Pagination
                size="small"
                count={pageCount}
                page={activePage}
                onChange={(_, value) => setPage({ number: value, filters })}
                getItemAriaLabel={(type, value) => (
                  type === 'page' ? `Go to failure page ${value}` : `Go to ${type} failure page`
                )}
              />
            )}
          </Stack>
        </Paper>
      )}
    </Stack>
  );
}
