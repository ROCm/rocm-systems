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
  return <Box component="details" data-testid="catalog-workloads" sx={{ mt: 1, fontSize: 12 }}>
    <Box component="summary" sx={{ cursor: 'pointer', py: 0.25, minHeight: 24 }}>Catalog workloads details</Box>
    <Typography variant="caption" color="text.secondary">Membership across all displayed benchmark sets; plotted cohorts also depend on configuration, suites and range.</Typography>
    <Box tabIndex={0} role="region" aria-label="Catalog workload membership" sx={{ overflowX: 'auto', maxHeight: 320, mt: 1 }}>
      <Table size="small" sx={{ minWidth: 420 }}><TableHead><TableRow><TableCell>Benchmark</TableCell><TableCell>Shared across benchmark sets</TableCell>{catalogs.map(({ catalog, label }) => <TableCell key={catalog.id} title="Latest published develop commit using this benchmark set (UTC)" sx={{ overflowWrap: 'anywhere' }}>{label}</TableCell>)}</TableRow></TableHead>
        <TableBody>{entries.map(({ test, shared, present }) => <TableRow key={test.id}><TableCell><Typography sx={{ fontSize: 12 }}>{test.name}</Typography><Typography variant="caption" color="text.secondary">{test.suite}</Typography></TableCell><TableCell aria-label={shared ? 'Shared' : 'Not shared'}>{shared ? '✓' : '—'}</TableCell>{present.map((value, index) => <TableCell key={catalogs[index].catalog.id} aria-label={value ? 'Member' : 'Not a member'}>{value ? '✓' : '—'}</TableCell>)}</TableRow>)}</TableBody>
      </Table>
    </Box>
    <Box sx={{ mt: 1 }}>{projection.rows.map((row) => <Typography key={row.key} variant="caption" component="p">{row.target} {row.mode}: {row.cohortIds.length} compared workloads · reference {row.reference?.runId ?? 'unavailable'} · latest {row.latest.runId ?? 'unavailable'}{row.latest.reason ? ` · ${row.latest.reason}` : ''}</Typography>)}</Box>
  </Box>;
}
