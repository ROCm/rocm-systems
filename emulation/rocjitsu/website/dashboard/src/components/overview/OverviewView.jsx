import { Box, Stack } from '@mui/material';
import DurationHistory from './DurationHistory';
import LargestChanges from './LargestChanges';
import MetricsGrid from './MetricsGrid';
import RecentRuns from './RecentRuns';


export default function OverviewView({
  viewModel,
  data,
  state,
  onCompareRun,
  onExploreRun,
  onOpenBenchmarks,
}) {
  return (
    <Stack sx={{ gap: 1.75, minWidth: 0 }}>
      <MetricsGrid metrics={viewModel.metrics} range={state.historyRange} />
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1.7fr) minmax(0, 1fr)' }, gap: 1.75, alignItems: 'stretch' }}>
        <DurationHistory
          history={viewModel.history}
          filters={state.filters}
          coverage={viewModel.metrics}
          range={state.historyRange}
          onRangeChange={state.setHistoryRange}
          onOpenBenchmarks={onOpenBenchmarks}
        />
        <LargestChanges
          changes={viewModel.changes}
          candidate={viewModel.history.latestRun}
          baseline={viewModel.history.firstRun}
        />
      </Box>
      <RecentRuns data={data} filters={state.filters} onCompareRun={onCompareRun} onExploreRun={onExploreRun} />
    </Stack>
  );
}
