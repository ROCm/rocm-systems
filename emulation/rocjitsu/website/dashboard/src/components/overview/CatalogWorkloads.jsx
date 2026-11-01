import { tableStyles } from '../../theme/tableStyles';
import { fontSizes, fontWeights } from '../../theme/typography';
import { Box, Table, TableBody, TableCell, TableHead, TableRow, Typography } from '@mui/material';

export default function CatalogWorkloads({ catalogs, projection }) {
  const definitions = new Map();
  const memberships = catalogs.map(({ catalog }) => {
    for (const test of catalog.tests) definitions.set(test.id, test);
    return new Set(Object.values(catalog.configurations).flat());
  });
  const entries = [...definitions.values()].map((test) => {
    const present = memberships.map((members) => members.has(test.id));
    return { test, present, shared: present.length > 0 && present.every(Boolean) };
  }).sort((a, b) => Number(b.shared) - Number(a.shared)
    || a.present.reduce((order, value, index) => order || Number(b.present[index]) - Number(value), 0)
    || a.test.name.localeCompare(b.test.name) || a.test.id.localeCompare(b.test.id));
  return <Box component="details" data-testid="catalog-workloads" sx={{ mt: 1, fontSize: fontSizes.caption }}>
    <Box component="summary" sx={{ cursor: 'pointer', py: '2px', minHeight: 24, color: 'text.primary', fontWeight: fontWeights.medium }}>Catalog workloads details</Box>
    <Typography variant="caption" color="text.primary" sx={{ lineHeight: '18px' }}>Membership across all displayed benchmark sets; plotted cohorts also depend on configuration, suites and range.</Typography>
    <Box tabIndex={0} role="region" aria-label="Catalog workload membership" sx={{ overflowX: 'auto', maxHeight: 320, mt: 1 }}>
      <Table size="small" aria-label="Catalog workload membership" sx={{ ...tableStyles.compact, minWidth: 420 }}><TableHead><TableRow><TableCell>Benchmark</TableCell><TableCell>Shared across benchmark sets</TableCell>{catalogs.map(({ catalog, label }) => <TableCell key={catalog.id} sx={{ overflowWrap: 'anywhere' }}>{label}</TableCell>)}</TableRow></TableHead>
        <TableBody>{entries.map(({ test, shared, present }) => <TableRow key={test.id}><TableCell><Typography sx={{ fontSize: fontSizes.body }}>{test.name}</Typography><Typography variant="caption" color="text.secondary">{test.suite}</Typography></TableCell><TableCell aria-label={shared ? 'Shared' : 'Not shared'}>{shared ? '✓' : '—'}</TableCell>{present.map((value, index) => <TableCell key={catalogs[index].catalog.id} aria-label={value ? 'Member' : 'Not a member'}>{value ? '✓' : '—'}</TableCell>)}</TableRow>)}</TableBody>
      </Table>
    </Box>
    <Box sx={{ mt: 2, pt: 1.5, borderTop: 1, borderColor: 'divider' }}>
      <Typography variant="caption" component="p" color="text.primary" sx={{ fontWeight: fontWeights.medium }}>Comparable workload:</Typography>
      {projection.rows.map((row) => <Typography key={row.key} variant="caption" component="p">{row.target} {row.mode}: {row.cohortIds.length}{row.latest.reason ? ` · ${row.latest.reason}` : ''}</Typography>)}
    </Box>
  </Box>;
}
