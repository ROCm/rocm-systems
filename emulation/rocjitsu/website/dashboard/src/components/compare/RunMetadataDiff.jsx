import { Box, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material';
import { runEnvironmentDetails } from '../../data/provenance';
import { shortSha } from '../../utils/formatters';
import { monoFont } from '../../theme/tokens';

function coverage(run, filters) {
  if (!run) return 'No run selected';
  const tests = run.tests.filter((test) => filters.targets.includes(test.target) && filters.suites.includes(test.suite) && (filters.modes ?? []).includes(test.mode));
  if (!tests.length) return 'No tests in selected scope';
  const completed = tests.filter((test) => test.status === 'completed' && Number.isFinite(test.durationSeconds)).length;
  return `${completed}/${tests.length} completed`;
}

function display(value) {
  if (value == null) return 'Not provided';
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

export default function RunMetadataDiff({ baseline, candidate, filters }) {
  const oldDetails = runEnvironmentDetails(baseline);
  const newDetails = runEnvironmentDetails(candidate);
  const keys = [...new Set([...oldDetails.keys(), ...newDetails.keys()])].sort();
  const facts = [
    ['coverage', 'Selected coverage', (run) => coverage(run, filters)],
    ['catalog', 'Test catalog', (run) => run?.testCatalog ?? run?.catalogId],
    ['machine', 'Machine', (run) => run?.machineId],
    ['configurations', 'Published configurations', (run) => run?.configurations],
  ].map(([key, label, value]) => ({ key, label, baseline: value(baseline), candidate: value(candidate) }));
  const environment = keys.map((key) => ({ key, label: oldDetails.get(key)?.label ?? newDetails.get(key)?.label, baseline: oldDetails.get(key)?.value, candidate: newDetails.get(key)?.value }));
  const rows = (items) => items.map((row) => {
    const changed = Boolean(baseline && candidate) && JSON.stringify(row.baseline ?? null) !== JSON.stringify(row.candidate ?? null);
    return <TableRow key={row.key} data-testid={`metadata-row-${row.key}`} data-different={changed} sx={changed ? { '& > .MuiTableCell-root': { bgcolor: 'action.selected', color: 'text.primary' } } : undefined}>
      <TableCell component="th" scope="row" sx={{ width: '26%', fontFamily: 'inherit', fontWeight: 500 }}>
        {row.label ?? row.key}
      </TableCell>
      {['baseline', 'candidate'].map((side) => <TableCell key={side} sx={{ width: '37%', fontFamily: monoFont, borderLeft: 1, borderColor: 'divider', overflowWrap: 'anywhere', whiteSpace: 'pre-wrap' }}>{display(row[side])}</TableCell>)}
    </TableRow>;
  });
  return <>
    <TableContainer data-testid="comparison-metadata-scroll" tabIndex={0} role="region" aria-label="Compared run metadata scroll area" sx={{ height: 366, overflow: 'auto', maxWidth: '100%', border: 1, borderColor: 'divider', borderRadius: 1, minWidth: 0, scrollbarGutter: 'stable', overscrollBehavior: 'contain', '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: 2 } }}>
      <Table stickyHeader size="small" aria-label="Run metadata differences" sx={{ tableLayout: 'fixed', minWidth: 720, borderCollapse: 'separate', borderSpacing: 0, '& th, & td': { p: 1.5, verticalAlign: 'top', fontSize: 12, lineHeight: '19px', overflowWrap: 'anywhere' }, '& thead th': { bgcolor: 'background.paper', fontSize: 11 } }}>
        <TableHead><TableRow>
          <TableCell sx={{ width: '26%' }}>Field</TableCell>
          {[[baseline, 'Baseline'], [candidate, 'Candidate']].map(([run, label]) => <TableCell key={label} data-testid={`${label.toLowerCase()}-run-information`} sx={{ width: '37%' }}>{label} run · {run ? shortSha(run) : 'Not selected'}<Box sx={{ mt: 0.25, fontFamily: monoFont, fontSize: 10 }}>{run?.runId}</Box></TableCell>)}
        </TableRow></TableHead>
        <TableBody>
          <TableRow><TableCell colSpan={3} sx={{ bgcolor: 'action.hover', fontFamily: 'inherit', fontWeight: 600 }}>Benchmark configuration</TableCell></TableRow>
          {rows(facts)}
          <TableRow><TableCell colSpan={3} sx={{ bgcolor: 'action.hover', fontFamily: 'inherit', fontWeight: 600 }}>Environment</TableCell></TableRow>
          {environment.length ? rows(environment) : <TableRow><TableCell colSpan={3}>No environment details provided.</TableCell></TableRow>}
        </TableBody>
      </Table>
    </TableContainer>
    <Stack direction="row" sx={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mt: 1 }}><Typography variant="caption" color="text.secondary">{baseline && candidate ? 'Highlighted rows indicate metadata differences, not performance.' : 'Select both runs to inspect metadata differences.'}</Typography><Typography variant="caption" color="text.secondary">Scroll to inspect all fields ↓</Typography></Stack>
  </>;
}
