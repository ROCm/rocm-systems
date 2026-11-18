import { tableStyles } from '../../theme/tableStyles';
import { fontSizes } from '../../theme/typography';
import Disclosure from '../shared/Disclosure';
import ScrollTableContainer from '../shared/ScrollTableContainer';
import { Table, TableBody, TableCell, TableHead, TableRow, Typography } from '@mui/material';

export default function CatalogWorkloads({ catalogs }) {
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
  return <Disclosure data-testid="catalog-workloads" title="Catalog workloads details">
    <Typography variant="body2" color="text.secondary" sx={{ mb: '12px' }}>Membership across all displayed benchmark sets; plotted cohorts also depend on configuration, suites and range.</Typography>
    <ScrollTableContainer tabIndex={0} role="region" aria-label="Catalog workload membership" sx={{ maxHeight: 320 }}>
      <Table size="small" stickyHeader aria-label="Catalog workload membership" sx={{ ...tableStyles.normal, minWidth: 420 }}><TableHead><TableRow><TableCell>Benchmark</TableCell><TableCell>Shared across benchmark sets</TableCell>{catalogs.map(({ catalog, label }) => <TableCell key={catalog.id} sx={{ overflowWrap: 'anywhere' }}>{label}</TableCell>)}</TableRow></TableHead>
        <TableBody>{entries.map(({ test, shared, present }) => <TableRow key={test.id}><TableCell><Typography sx={{ fontSize: fontSizes.body }}>{test.name}</Typography><Typography variant="caption" color="text.secondary">{test.suite}</Typography></TableCell><TableCell aria-label={shared ? 'Shared' : 'Not shared'}>{shared ? '✓' : '—'}</TableCell>{present.map((value, index) => <TableCell key={catalogs[index].catalog.id} aria-label={value ? 'Member' : 'Not a member'}>{value ? '✓' : '—'}</TableCell>)}</TableRow>)}</TableBody>
      </Table>
    </ScrollTableContainer>
  </Disclosure>;
}
