import { Box, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material';
import { safeExternalUrl } from './branchPresentation';
import { monoFont } from '../../theme/tokens';

function display(value) {
  if (value == null) return 'Not provided';
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

export default function BranchEnvironment({ candidate, reference, selection, repository }) {
  const baselineFacts = new Map((reference?.provenance?.details ?? []).map((fact) => [fact.key, fact]));
  const candidateFacts = new Map((candidate?.provenance?.details ?? []).map((fact) => [fact.key, fact]));
  const environment = [...new Set([...baselineFacts.keys(), ...candidateFacts.keys()])].sort().map((key) => ({
    key, label: candidateFacts.get(key)?.label ?? baselineFacts.get(key)?.label,
    baseline: baselineFacts.get(key)?.value, candidate: candidateFacts.get(key)?.value,
    baselineLabel: baselineFacts.get(key)?.label, candidateLabel: candidateFacts.get(key)?.label,
  }));
  const configuration = (run) => run?.configurations.find((item) => item.target === selection.target && item.threadingMode === selection.mode);
  const sourceFacts = [
    ['attempt', 'Exact attempt', (run) => run?.runId], ['branch', 'Source branch', (run) => run?.branch],
    ['commit', 'Source SHA', (run) => run?.provenance?.rocjitsuCommitSha], ['message', 'Commit message', (run) => run?.provenance?.commitMessage],
    ['baseBranch', 'Recorded base branch', (run) => run?.sourceBase?.branch], ['baseCommit', 'Recorded base SHA', (run) => run?.sourceBase?.commit],
    ['committedAt', 'Commit time (UTC)', (run) => run?.commitTimestamp], ['completedAt', 'Execution completed (UTC)', (run) => run?.timestamp],
    ['machine', 'Machine', (run) => run?.machineId], ['trigger', 'Trigger', (run) => run?.trigger],
    ['catalog', 'Test catalog', (run) => run?.catalogId], ['environment', 'Environment identity', (run) => run?.environmentId],
    ['configuration', 'Selected configuration', (run) => run ? configuration(run) ? `${selection.target} · ${selection.mode}` : 'Not published' : null],
  ].map(([key, label, value]) => ({ key, label, baseline: value(reference), candidate: value(candidate) }));
  const rows = (items, prefix) => items.map((fact) => {
    const changed = Boolean(reference && candidate) && JSON.stringify(fact.baseline ?? null) !== JSON.stringify(fact.candidate ?? null);
    return <TableRow key={fact.key} data-testid={`branch-${prefix}-${fact.key}`} data-different={changed} sx={changed ? { '& > .MuiTableCell-root': { bgcolor: 'action.selected', color: 'text.primary' } } : undefined}>
      <TableCell component="th" scope="row" sx={{ width: '30%' }}>
        {fact.label ?? fact.key}
        {prefix === 'environment' && <Typography component="div" variant="caption" color="text.secondary" sx={{ fontFamily: monoFont }}>{fact.key}</Typography>}
        {fact.baselineLabel && fact.candidateLabel && fact.baselineLabel !== fact.candidateLabel && <Typography variant="caption">Reference label: {fact.baselineLabel}</Typography>}
      </TableCell>
      <TableCell sx={{ fontFamily: monoFont }}>{display(fact.baseline)}</TableCell>
      <TableCell sx={{ fontFamily: monoFont }}>{display(fact.candidate)}</TableCell>
    </TableRow>;
  });
  return <Box component="details" data-testid="branch-environment" sx={{ borderTop: 1, borderColor: 'divider', mt: 1 }}>
    <Box component="summary" sx={{ cursor: 'pointer', minHeight: 44, display: 'list-item', py: 1.5, ml: 2 }}>Environment &amp; measurement details</Box>
    <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>These are independent published attempts. Matching workloads does not imply equivalent environments or machines.</Typography>
    <TableContainer role="region" tabIndex={0} aria-label="Branch source and environment facts, scrollable" sx={{ maxHeight: 366, scrollbarGutter: 'stable', overscrollBehavior: 'contain', border: 1, borderColor: 'divider' }}>
      <Table size="small" stickyHeader aria-label="Branch source and environment facts" sx={{ tableLayout: 'fixed', minWidth: 540, '& td, & th': { p: 1.25, overflowWrap: 'anywhere', whiteSpace: 'pre-wrap', verticalAlign: 'top' } }}>
        <TableHead><TableRow><TableCell>Field</TableCell><TableCell>Reference · {reference?.runId ?? 'Unavailable'}</TableCell><TableCell>Candidate · {candidate?.runId ?? 'Unavailable'}</TableCell></TableRow></TableHead>
        <TableBody>
          {rows(sourceFacts, 'source')}
          <TableRow><TableCell colSpan={3} sx={{ bgcolor: 'action.hover', fontWeight: 600 }}>Published environment facts</TableCell></TableRow>
          {environment.length ? rows(environment, 'environment') : <TableRow><TableCell colSpan={3}>No environment facts provided.</TableCell></TableRow>}
        </TableBody>
      </Table>
    </TableContainer>
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
      {safeExternalUrl(repository) && <Button component="a" href={repository} target="_blank" rel="noopener noreferrer" sx={{ minHeight: 44 }}>Source repository ↗</Button>}
    </Box>
  </Box>;
}
