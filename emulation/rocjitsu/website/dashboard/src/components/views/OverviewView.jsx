import { useMemo } from 'react';
import { Box, Stack } from '@mui/material';
import { hasCommitTimestamp, sortRunsByCommit } from '../../data/runOrdering';
import { formatShortDate } from '../../utils/formatters';
import { isValidTimestamp } from '../../utils/timestamp';
import DurationHistory from '../overview/DurationHistory';
import LargestChanges from '../overview/LargestChanges';
import MetricsGrid from '../overview/MetricsGrid';
import RecentRuns from '../overview/RecentRuns';

export default function OverviewView({ viewModel, data, state }) {
  const catalogs = useMemo(() => {
    const latest = new Map();
    for (const run of [...data.runs.filter((entry) => !hasCommitTimestamp(entry)), ...sortRunsByCommit(data.runs)]) {
      const catalog = data.catalogs?.[run.catalogId];
      if (catalog) latest.set(catalog.id, { catalog, timestamp: run.commitTimestamp });
    }
    return [...latest.values()].sort((a, b) => (isValidTimestamp(b.timestamp) ? Date.parse(b.timestamp) : -Infinity) - (isValidTimestamp(a.timestamp) ? Date.parse(a.timestamp) : -Infinity) || a.catalog.id.localeCompare(b.catalog.id))
      .map((entry) => ({ ...entry, label: `${entry.catalog.id} · ${formatShortDate(entry.timestamp)}` }));
  }, [data]);
  return <Stack sx={{ gap: 2, minWidth: 0 }}>
    <MetricsGrid projection={viewModel} />
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'minmax(0,1.9fr) minmax(0,1fr)' }, gap: 2, alignItems: 'stretch' }}>
      <DurationHistory projection={viewModel} catalogs={catalogs} onRangeChange={state.setHistoryRange} onCatalogChange={state.setOverviewCatalogId} />
      <LargestChanges projection={viewModel} catalogs={catalogs} />
    </Box>
    <RecentRuns data={data} filters={state.filters} />
  </Stack>;
}
