import { Box, Stack, Typography } from '@mui/material';
import DurationHistory from './DurationHistory';
import LargestChanges from './LargestChanges';
import MetricsGrid from './MetricsGrid';
import RecentRuns from './RecentRuns';
import ResultsTable from './ResultsTable';

export default function OverviewView({ viewModel, data, state, onCompareRun, onExploreRun }) {
  return (
    <Stack sx={{ gap: 1.75 }}>
      <MetricsGrid metrics={viewModel.metrics} candidate={viewModel.candidate} baseline={viewModel.baseline} />
      <Typography variant="caption" color="text.secondary" sx={{ px: 0.5 }}>
        <Box component="span" color="text.primary" fontWeight={750}>Overview scope: </Box>
        newest attempt of the latest commit. Its baseline is the newest completed attempt of the nearest earlier commit.
      </Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1.55fr) minmax(350px, .85fr)' }, gap: 1.75, alignItems: 'stretch' }}>
        <DurationHistory
          history={viewModel.history}
          range={state.historyRange}
          onRangeChange={state.setHistoryRange}
        />
        <LargestChanges changes={viewModel.changes} candidate={viewModel.candidate} baseline={viewModel.baseline} />
      </Box>
      <ResultsTable
        results={viewModel.results}
        run={viewModel.candidate}
        baseline={viewModel.baseline}
        repository={data.repository}
        search={state.search}
        onSearch={state.setSearch}
      />
      <RecentRuns data={data} filters={state.filters} onCompareRun={onCompareRun} onExploreRun={onExploreRun} />
    </Stack>
  );
}
