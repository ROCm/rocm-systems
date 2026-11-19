import { fontSizes, fontWeights } from '../../theme/typography';
import { Box, Stack, Typography } from '@mui/material';
import SectionCard from '../shared/SectionCard';
import CategoryTag from '../shared/CategoryTag';
import { shapeTokens } from '../../theme/tokens';
import { formatPercent } from '../../utils/formatters';
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
  return <SectionCard title="Largest changes" subtitle="Duration change (%) · lower is faster" sx={{ height: '100%', minWidth: 0 }} data-testid="largest-changes">
    {!changes.length ? <Typography color="text.secondary" sx={{ py: '40px', px: '12px', fontSize: fontSizes.body, textAlign: 'center' }}>{!projection.rows.length ? 'No configurations selected.' : `No comparable results: ${unavailable.join('; ')}.`}</Typography> : changes.map(({ id, result, delta, row }, index) => {
      const state = classifyDurationChange(delta);
      const tone = changeTone(state);
      return <Box key={`${row.key}/${id}`} data-configuration={row.key} data-workload={id} data-change-state={state} sx={{ py: shapeTokens.rowPadding, ...(index === 0 && { pt: 0, mt: '-2px' }) }}>
        <Box sx={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto', alignItems: 'center', columnGap: '10px', rowGap: '4px' }}>
          <Box sx={{ minWidth: 0 }}><Typography sx={{ fontWeight: fontWeights.medium, overflowWrap: 'anywhere', fontSize: fontSizes.body }}>{result.name}</Typography>
            <Stack direction="row" sx={{ mt: '3px', gap: '5px', flexWrap: 'wrap' }}><CategoryTag subdued kind="target" label={row.target} /><CategoryTag subdued label={result.suite} /><CategoryTag subdued kind="mode" label={row.mode} /></Stack>
          </Box>
          <Typography variant="numeric" sx={{ flexShrink: 0, color: tone === 'neutral' ? 'text.secondary' : `${tone}.main` }}>{formatPercent(delta)}</Typography>
        </Box>
        {/* The line is a border: the browser snaps borders to a whole number of device pixels, so every
            bar paints the same thickness at any row position. */}
        <Box aria-hidden="true" sx={{ '--bar': '2px', position: 'relative', mt: 1, height: 0, borderTop: 'var(--bar) solid', borderColor: 'dashboard.barTrack' }}>
          <Box data-testid="change-magnitude-bar" sx={{ position: 'absolute', top: 'calc(-1 * var(--bar))', left: 0, height: 0, width: `${maximum ? Math.abs(delta) / maximum * 100 : 0}%`, borderTop: 'var(--bar) solid', borderColor: tone === 'neutral' ? 'text.secondary' : `${tone}.main` }} />
        </Box>
      </Box>;
    })}
  </SectionCard>;
}
