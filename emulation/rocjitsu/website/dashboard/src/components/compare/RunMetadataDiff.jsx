import { tableStyles, changedRowSx } from '../../theme/tableStyles';
import { fontSizes, fontWeights, sansFont } from '../../theme/typography';
import { Fragment } from 'react';
import { groupMetadataFacts, metadataLabel } from '../shared/metadataPresentation';
import { Box, Stack, Table, TableBody, TableCell, TableHead, TableRow, Typography } from '@mui/material';
import { runEnvironmentDetails } from '../../data/provenance';
import { formatMetadataValue, shortSha } from '../../utils/formatters';
import ScrollTableContainer from '../shared/ScrollTableContainer';

function coverage(run, filters) {
  if (!run) return 'No run selected';
  const tests = run.tests.filter((test) => filters.targets.includes(test.target) && filters.suites.includes(test.suite) && (filters.modes ?? []).includes(test.mode));
  if (!tests.length) return 'No tests in selected scope';
  const completed = tests.filter((test) => test.status === 'completed' && Number.isFinite(test.durationSeconds)).length;
  return `${completed}/${tests.length} completed`;
}

export default function RunMetadataDiff({ baseline, candidate, filters }) {
  const oldDetails = runEnvironmentDetails(baseline);
  const newDetails = runEnvironmentDetails(candidate);
  const keys = [...new Set([...oldDetails.keys(), ...newDetails.keys()])].sort();
  const facts = [
    ['coverage', 'Selected coverage', (run) => coverage(run, filters)],
    ['catalog', 'Benchmark set', (run) => run?.catalogId],
    ['machine', 'Machine', (run) => run?.machineId],
    ['configurations', 'Published configurations', (run) => run?.configurations],
  ].map(([key, label, value]) => ({ key, label, baseline: value(baseline), candidate: value(candidate) }));
  const environment = keys.map((key) => ({ key, label: oldDetails.get(key)?.label ?? newDetails.get(key)?.label, baseline: oldDetails.get(key)?.value, candidate: newDetails.get(key)?.value }));
  const rows = (items) => items.map((row) => {
    const changed = Boolean(baseline && candidate) && JSON.stringify(row.baseline ?? null) !== JSON.stringify(row.candidate ?? null);
    return <TableRow key={row.key} data-testid={`metadata-row-${row.key}`} data-different={changed} sx={changed ? changedRowSx : undefined}>
      <TableCell component="th" scope="row" sx={{ width: '26%', fontFamily: 'inherit', fontWeight: fontWeights.medium }}>
        {metadataLabel(row.label, row.key)}
      </TableCell>
      {['baseline', 'candidate'].map((side) => <TableCell key={side} sx={{ width: '37%', fontFamily: sansFont, borderLeft: 1, borderColor: 'divider', overflowWrap: 'anywhere', whiteSpace: 'pre-wrap' }}>{row.key === 'configurations' && Array.isArray(row[side])
        ? row[side].length ? <Box component="ul" sx={{ m: 0, pl: 2 }}>{row[side].map((configuration) => <li key={`${configuration.target}:${configuration.thread_policy}`}>{configuration.target} · {configuration.thread_policy}</li>)}</Box> : 'None published'
        : formatMetadataValue(row[side])}</TableCell>)}
    </TableRow>;
  });
  return <>
    <ScrollTableContainer data-testid="comparison-metadata-scroll" tabIndex={0} role="region" aria-label="Compared run metadata scroll area" sx={{ maxHeight: 400, minWidth: 0, scrollbarGutter: 'stable', overscrollBehavior: 'contain' }}>
      <Table stickyHeader size="small" aria-label="Run metadata differences" sx={{ ...tableStyles.normal, tableLayout: 'fixed', minWidth: 720, borderCollapse: 'separate', borderSpacing: 0, '& th, & td': { verticalAlign: 'top', overflowWrap: 'anywhere' }, '& thead th': { bgcolor: 'dashboard.tableHeader' } }}>
        <TableHead><TableRow>
          <TableCell sx={{ width: '26%' }}>Field</TableCell>
          {[[baseline, 'Baseline'], [candidate, 'Candidate']].map(([run, label]) => <TableCell key={label} data-testid={`${label.toLowerCase()}-run-information`} sx={{ width: '37%' }}>{label} run · {run ? shortSha(run) : 'Not selected'}<Box sx={{ mt: '2px', fontFamily: sansFont, fontSize: fontSizes.caption }}>{run?.runId}</Box></TableCell>)}
        </TableRow></TableHead>
        <TableBody>
          <TableRow><TableCell colSpan={3} sx={{ bgcolor: 'dashboard.tableGroup', color: 'text.secondary', fontFamily: 'inherit', fontWeight: fontWeights.medium }}>Benchmark configuration</TableCell></TableRow>
          {rows(facts)}
          {environment.length ? groupMetadataFacts(environment).map((group) => <Fragment key={group.label}>
            <TableRow><TableCell component="th" scope="rowgroup" colSpan={3} sx={{ bgcolor: 'dashboard.tableGroup', color: 'text.secondary', fontFamily: 'inherit', fontWeight: fontWeights.medium }}>{group.label}</TableCell></TableRow>
            {rows(group.rows)}
          </Fragment>) : <TableRow><TableCell colSpan={3}>No environment details provided.</TableCell></TableRow>}
        </TableBody>
      </Table>
    </ScrollTableContainer>
    <Stack direction="row" sx={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mt: 1 }}><Typography variant="caption" color="text.muted">{baseline && candidate ? 'Highlighted rows indicate metadata differences, not performance.' : 'Select both runs to inspect metadata differences.'}</Typography><Typography variant="caption" color="text.muted">Scroll to inspect all fields ↓</Typography></Stack>
  </>;
}
