import { Box, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material';
import { Fragment } from 'react';
import { runEnvironmentDetails } from '../../data/provenance';
import { groupMetadataFacts, metadataLabel } from '../shared/metadataPresentation';
import { safeExternalUrl } from './branchPresentation';
import { monoFont } from '../../theme/tokens';
import { formatMetadataValue } from '../../utils/formatters';

export default function BranchEnvironment({ candidate, reference, selection, repository }) {
  const baselineFacts = runEnvironmentDetails(reference);
  const candidateFacts = runEnvironmentDetails(candidate);
  const environment = [...new Set([...baselineFacts.keys(), ...candidateFacts.keys()])].sort().map((key) => ({
    key, label: candidateFacts.get(key)?.label ?? baselineFacts.get(key)?.label,
    baseline: baselineFacts.get(key)?.value, candidate: candidateFacts.get(key)?.value,
    baselineLabel: baselineFacts.get(key)?.label, candidateLabel: candidateFacts.get(key)?.label,
  }));
  const configuration = (run) => run?.configurations.find((item) => item.target === selection.target && item.thread_policy === selection.mode);
  const sourceGroups = [
    ['Benchmark configuration', [
      ['catalog', 'Benchmark set', (run) => run?.catalogId],
      ['configuration', 'Selected configuration', (run) => run ? configuration(run) ? `${selection.target} · ${selection.mode}` : 'Not published' : null],
    ]],
    ['Source & base revision', [
      ['attempt', 'Exact attempt', (run) => run?.runId], ['branch', 'Source branch', (run) => run?.branch],
      ['commit', 'Source SHA', (run) => run?.provenance?.rocjitsuCommitSha], ['message', 'Commit message', (run) => run?.provenance?.commitMessage],
      ['baseBranch', 'Recorded base branch', (run) => run?.sourceBase?.branch], ['baseCommit', 'Recorded base SHA', (run) => run?.sourceBase?.commit],
    ]],
    ['Execution', [
      ['committedAt', 'Commit time (UTC)', (run) => run?.commitTimestamp], ['completedAt', 'Execution completed (UTC)', (run) => run?.timestamp],
      ['machine', 'Machine', (run) => run?.machineId], ['trigger', 'Trigger', (run) => run?.trigger],
    ]],
  ].map(([label, fields]) => ({ label, facts: fields.map(([key, label, value]) => ({ key, label, baseline: value(reference), candidate: value(candidate) })) }));
  const rows = (items, prefix) => items.map((fact) => {
    const changed = Boolean(reference && candidate) && JSON.stringify(fact.baseline ?? null) !== JSON.stringify(fact.candidate ?? null);
    return <TableRow key={fact.key} data-testid={`branch-${prefix}-${fact.key}`} data-different={changed} sx={changed ? { '& > .MuiTableCell-root': { bgcolor: 'action.selected', color: 'text.primary' } } : undefined}>
      <TableCell component="th" scope="row" sx={{ width: '30%' }}>
        {metadataLabel(fact.label, fact.key)}
        {fact.baselineLabel && fact.candidateLabel && fact.baselineLabel !== fact.candidateLabel && <Typography variant="caption">Reference label: {fact.baselineLabel}</Typography>}
      </TableCell>
      <TableCell sx={{ fontFamily: monoFont }}>{formatMetadataValue(fact.baseline)}</TableCell>
      <TableCell sx={{ fontFamily: monoFont }}>{formatMetadataValue(fact.candidate)}</TableCell>
    </TableRow>;
  });
  return <Box component="details" data-testid="branch-environment" sx={{ borderTop: 1, borderColor: 'divider', mt: 1 }}>
    <Box component="summary" sx={{ cursor: 'pointer', minHeight: 44, display: 'list-item', py: 1.5, ml: 2 }}>Environment &amp; measurement details</Box>
    <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>These are independent published attempts. Matching workloads does not imply equivalent environments or machines. Highlighted rows indicate metadata differences, not performance.</Typography>
    <TableContainer role="region" tabIndex={0} aria-label="Branch source and environment facts, scrollable" sx={{ maxHeight: 366, scrollbarGutter: 'stable', overscrollBehavior: 'contain', border: 1, borderColor: 'divider' }}>
      <Table size="small" stickyHeader aria-label="Branch source and environment facts" sx={{ tableLayout: 'fixed', minWidth: 540, '& td, & th': { p: 1.25, overflowWrap: 'anywhere', whiteSpace: 'pre-wrap', verticalAlign: 'top' } }}>
        <TableHead><TableRow><TableCell sx={{ width: '30%' }}>Field</TableCell><TableCell sx={{ width: '35%' }}>Reference · {reference?.runId ?? 'Unavailable'}</TableCell><TableCell sx={{ width: '35%' }}>Candidate · {candidate?.runId ?? 'Unavailable'}</TableCell></TableRow></TableHead>
        <TableBody>
          {sourceGroups.map(({ label, facts }) => <Fragment key={label}>
            <TableRow><TableCell component="th" scope="rowgroup" colSpan={3} sx={{ bgcolor: 'action.hover', fontWeight: 600 }}>{label}</TableCell></TableRow>
            {rows(facts, 'source')}
          </Fragment>)}
          {environment.length ? groupMetadataFacts(environment).map((group) => <Fragment key={group.label}>
            <TableRow><TableCell component="th" scope="rowgroup" colSpan={3} sx={{ bgcolor: 'action.hover', fontWeight: 600 }}>{group.label}</TableCell></TableRow>
            {rows(group.rows, 'environment')}
          </Fragment>) : <TableRow><TableCell colSpan={3}>No environment facts provided.</TableCell></TableRow>}
        </TableBody>
      </Table>
    </TableContainer>
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
      {safeExternalUrl(repository) && <Button component="a" href={repository} target="_blank" rel="noopener noreferrer" sx={{ minHeight: 44 }}>Source repository ↗</Button>}
    </Box>
  </Box>;
}
