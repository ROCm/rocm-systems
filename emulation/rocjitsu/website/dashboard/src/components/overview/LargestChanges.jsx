import { Box, Stack, Typography } from '@mui/material';
import SectionCard from '../shared/SectionCard';
import CategoryTag from '../shared/CategoryTag';
import { monoFont } from '../../theme/tokens';
import { formatPercent, shortSha } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';

export default function LargestChanges({ projection, catalogs = [] }) {
  // Match the disclosure's all-displayed-catalog membership, not only catalogs
  // represented by runs inside the current history window.
  const memberships = catalogs.map(({ catalog }) => new Set(Object.values(catalog.configurations).flat()));
  const sharedIds = new Set([...(memberships[0] ?? [])].filter((id) => memberships.every((members) => members.has(id))));
  const sharedOnly = projection.catalogId == null;
  const changes = projection.rows.flatMap((row) => {
    const baseline = row.reference;
    const candidate = row.latest;
    if (!(candidate?.durationSeconds > 0 && baseline?.durationSeconds > 0)) return [];
    let references;
    return candidate.results.flatMap(({ id, result }) => {
      if (sharedOnly && !sharedIds.has(id)) return [];
      if (!references) {
        references = new Map();
        for (const item of baseline.results) if (!references.has(item.id)) references.set(item.id, item.result);
      }
      const reference = references.get(id);
      const delta = reference?.durationSeconds > 0 ? (result.durationSeconds - reference.durationSeconds) / reference.durationSeconds * 100 : null;
      return Number.isFinite(delta) ? [{ id, result, delta, row, baseline, candidate }] : [];
    });
  }).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta) || a.id.localeCompare(b.id)).slice(0, 6);
  const maximum = Math.max(...changes.map(({ delta }) => Math.abs(delta)), 0);
  const unavailable = [...new Set(projection.rows.map((row) => row.cohortIds.length && (!sharedOnly || row.cohortIds.some((id) => sharedIds.has(id))) ? row.latest?.reason ?? row.reference?.reason ?? 'no valid reference' : 'No shared workloads'))];
  return <SectionCard title="Largest Changes" subtitle="Duration change (%) · lower is faster" sx={{ height: '100%', minWidth: 0 }} contentSx={{ p: 2 }} data-testid="largest-changes">
    {!changes.length ? <Typography color="text.secondary" sx={{ py: 4, fontSize: 12 }}>{!projection.rows.length ? 'No configurations selected.' : `No comparable results: ${unavailable.join('; ')}.`}</Typography> : changes.map(({ id, result, delta, row, baseline, candidate }) => {
      const state = classifyDurationChange(delta);
      const tone = changeTone(state);
      return <Box key={`${row.key}/${id}`} data-configuration={row.key} data-workload={id} data-change-state={state} sx={{ py: 1.1 }}>
        <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center', gap: 1 }}>
          <Box sx={{ minWidth: 0 }}><Typography sx={{ fontWeight: 600, overflowWrap: 'anywhere', fontSize: 12 }}>{result.name}</Typography>
            <Stack direction="row" sx={{ mt: 0.4, gap: 0.5, flexWrap: 'wrap' }}><CategoryTag kind="target" label={row.target} /><CategoryTag label={result.suite} /><CategoryTag kind="mode" label={row.mode} /></Stack>
          </Box>
          <Typography title={`Latest ${shortSha(candidate.run)} vs reference ${shortSha(baseline.run)}`} sx={{ flexShrink: 0, fontFamily: monoFont, fontSize: 15, color: tone === 'neutral' ? 'text.secondary' : `${tone}.main` }}>{formatPercent(delta)}</Typography>
        </Stack>
        <Box aria-hidden="true" sx={{ mt: 1, height: 3, bgcolor: 'divider', borderRadius: 1 }}><Box data-testid="change-magnitude-bar" sx={{ height: '100%', width: `${maximum ? Math.abs(delta) / maximum * 100 : 0}%`, bgcolor: tone === 'neutral' ? 'text.disabled' : `${tone}.main`, borderRadius: 1 }} /></Box>
      </Box>;
    })}
  </SectionCard>;
}
