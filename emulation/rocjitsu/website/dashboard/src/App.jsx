import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Container,
  CssBaseline,
  LinearProgress,
  Paper,
  Skeleton,
  Stack,
  ThemeProvider,
  Typography,
} from '@mui/material';
import DashboardHeader from './components/layout/DashboardHeader';
import DashboardShell from './components/layout/DashboardShell';
import OverviewView from './components/overview/OverviewView';
import BenchmarksView from './components/views/BenchmarksView';
import CompareRunsView from './components/views/CompareRunsView';
import BranchRunsView from './components/views/BranchRunsView';
import { isLoadCancelled, loadDashboardDataFiles } from './data/dashboardData';
import { summarizeDashboardDataError } from './data/dashboardDataError';
import { resolvePublishedDataUrls } from './data/publishedDataUrls';
import { DASHBOARD_SITE_CONFIG } from './config/siteConfig';
import { selectOverview } from './data/selectors';
import { useDashboardState } from './hooks/useDashboardState';
import { visuallyHiddenStyles } from './theme/styles';
import { createDashboardTheme } from './theme/theme';

const { indexUrl: dataIndexUrl } = resolvePublishedDataUrls();
const DATA_CACHE_GENERATION_STORAGE_KEY = 'rocjitsu-data-cache-generation';

function readCacheGeneration() {
  try {
    return window.localStorage.getItem(DATA_CACHE_GENERATION_STORAGE_KEY);
  } catch {
    return null;
  }
}

function saveCacheGeneration(cacheGeneration) {
  try {
    window.localStorage.setItem(DATA_CACHE_GENERATION_STORAGE_KEY, cacheGeneration);
  } catch {
    // Cache persistence is optional; a storage failure must not discard loaded data.
  }
}

function LoadingDataState({ progress }) {
  const determinate = progress.total > 0;
  return (
    <Box component="main" sx={{ minHeight: 'calc(100vh - 76px)' }}>
      <Container maxWidth={false} sx={{ maxWidth: 1600, px: { xs: 1.5, sm: 2, xl: 3 }, pt: 2, pb: 4 }}>
        <Paper
          data-testid="dashboard-data-loading"
          aria-busy="true"
          variant="outlined"
          sx={{ overflow: 'hidden', borderRadius: 3 }}
        >
          <Typography role="status" aria-live="polite" sx={visuallyHiddenStyles}>
            Loading benchmark run data
          </Typography>
          <LinearProgress
            aria-label="Loading benchmark run data"
            aria-valuetext={determinate
              ? `${progress.loaded} of ${progress.total} run files loaded`
              : 'Loading benchmark run data'}
            variant={determinate ? 'determinate' : 'indeterminate'}
            {...(determinate ? { value: (progress.loaded / progress.total) * 100 } : {})}
          />
          <Box sx={{ p: { xs: 2, sm: 2.5 } }}>
            <Typography fontWeight={700}>Loading benchmark run data…</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.4 }}>
              The dashboard shell is ready while run files are fetched and validated.
            </Typography>
            {determinate && (
              <Typography data-testid="dashboard-load-progress" variant="body2" sx={{ color: 'text.secondary', mt: 0.4 }}>
                {progress.loaded} of {progress.total} run files loaded
              </Typography>
            )}
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} sx={{ mt: 2.25 }}>
              <Skeleton variant="rounded" animation="wave" height={54} sx={{ flex: 1 }} />
              <Skeleton variant="rounded" animation="wave" height={54} sx={{ flex: 1 }} />
            </Stack>
          </Box>
        </Paper>
        <Skeleton variant="rounded" animation="wave" height={52} sx={{ mt: 1.75, borderRadius: 3 }} />
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, minmax(0, 1fr))' }, gap: 1.5, mt: 1.75 }}>
          {[0, 1, 2].map((index) => (
            <Skeleton key={index} variant="rounded" animation="wave" height={150} sx={{ borderRadius: 3 }} />
          ))}
        </Box>
      </Container>
    </Box>
  );
}

const emptyDashboardData = {
  schemaVersion: null,
  ...DASHBOARD_SITE_CONFIG,
  generatedAt: null,
  runs: [],
  allRuns: [],
  modes: [],
  testCatalog: [],
  targets: [],
  suites: [],
  latestRun: null,
  latestCommitRun: null,
  backfillRunIds: new Set(),
};

function emptyOverview(range) {
  return {
    candidate: null,
    baseline: null,
    changes: [],
    results: [],
    metrics: {
      duration: null,
      durationDelta: null,
      completed: 0,
      total: 0,
      failed: 0,
      completeness: null,
    },
    history: {
      range,
      mode: 'daily-by-commit',
      anchorDay: null,
      slots: [],
      dayKeys: [],
      axisMax: 0,
      currentDuration: null,
      firstRun: null,
      latestRun: null,
      durationDelta: null,
      summary: '—',
      series: [],
    },
  };
}

function Dashboard({ data, state, dataError = null, onRetry = null }) {
  const hasData = data.runs.length > 0;
  const dataErrorMessage = dataError ? summarizeDashboardDataError(dataError) : null;
  // Overview derives the whole history, so it stays uncomputed while another tab owns the view.
  const overview = useMemo(
    () => (state.tab === 'overview'
      ? hasData
        ? selectOverview(data, state.filters, state.historyRange)
        : emptyOverview(state.historyRange)
      : null),
    [data, hasData, state.filters, state.historyRange, state.tab],
  );
  const openBranchComparison = (selection) => {
    state.openComparison(selection);
    window.requestAnimationFrame(() => window.scrollTo({ top: 0, left: 0, behavior: 'instant' }));
  };
  const openBenchmarks = () => {
    state.setExplorerRunIds([]);
    state.setTab('benchmarks');
    window.requestAnimationFrame(() => window.scrollTo({ top: 0 }));
  };
  const selectExplorerRun = (runId) => {
    state.setExplorerRunIds((current) => (
      current.includes(runId)
        ? current.filter((candidateId) => candidateId !== runId)
        : [...current, runId]
    ));
  };

  return (
    <>
      <Box component="main" sx={{ minHeight: 'calc(100vh - 140px)' }}>
        <Container maxWidth={false} sx={{ maxWidth: 1600, px: { xs: 1.5, sm: 2, xl: 3 }, pt: 2, pb: 4 }}>

          {dataError && (
            <Alert
              data-testid="dashboard-data-error"
              severity="warning"
              variant="outlined"
              action={<Button color="inherit" size="small" onClick={onRetry}>Retry</Button>}
              sx={{ mb: 1.75 }}
            >
              <Typography fontWeight={700}>No available test data</Typography>
              {dataErrorMessage !== 'No available test data' && (hasData || !dataErrorMessage.startsWith('Dashboard schema 1 requires migration to schema 2')) && (
                <Typography variant="body2" sx={{ mt: 0.5, whiteSpace: 'pre-line' }}>
                  {dataErrorMessage}
                </Typography>
              )}
              <Typography variant="body2" sx={{ mt: 1 }}>Dashboard values will remain empty until benchmark data is published to the site.</Typography>
            </Alert>
          )}

          <Box role="tabpanel" id={`dashboard-panel-${state.tab}`} aria-labelledby={`dashboard-tab-${state.tab}`}>
          {state.tab === 'branch' && <BranchRunsView data={data} state={state} onOpenComparison={openBranchComparison} />}
          {state.tab === 'overview' && (
            <OverviewView
              viewModel={overview}
              data={data}
              state={state}
              onOpenBenchmarks={openBenchmarks}
            />
          )}
          {state.tab === 'benchmarks' && (
            <BenchmarksView
              data={data}
              filters={state.filters}
              historyRange={state.historyRange}
              onRangeChange={state.setHistoryRange}
              selectedBenchmarks={state.selectedBenchmarks}
              onBenchmarksChange={state.setSelectedBenchmarks}
              selectedRunIds={state.explorerRunIds}
              onSelectRun={selectExplorerRun}
              onClearSelectedRuns={() => state.setExplorerRunIds([])}
            />
          )}
          {state.tab === 'compare' && (
            <CompareRunsView
              data={data}
              filters={state.filters}
              selectedBaselineId={state.comparisonBaselineId}
              selectedCandidateId={state.comparisonCandidateId}
              onBaselineChange={state.setComparisonBaselineId}
              onCandidateChange={state.setComparisonCandidateId}
              onSwap={state.setComparisonPair}
            />
          )}
          </Box>
        </Container>
      </Box>
    </>
  );
}

function readPreferredMode() {
  try {
    const saved = window.localStorage.getItem('rocjitsu-color-mode');
    if (saved === 'light' || saved === 'dark') return saved;
  } catch { /* Theme persistence is optional. */ }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const [mode, setMode] = useState(readPreferredMode);
  const [dataState, setDataState] = useState({
    data: null,
    manifest: null,
    sourceData: null,
    error: null,
  });
  // Progress is tagged with the attempt that produced it so a retry starts from zero without an
  // extra state reset.
  const [progress, setProgress] = useState({ attempt: 0, loaded: 0, total: 0 });
  const [loadRequest, setLoadRequest] = useState(() => ({
    attempt: 0,
    reloadAll: false,
    cacheGeneration: readCacheGeneration(),
  }));
  const theme = useMemo(() => createDashboardTheme(mode), [mode]);
  const state = useDashboardState(dataState.data ?? emptyDashboardData);

  useEffect(() => {
    const controller = new AbortController();
    loadDashboardDataFiles({
      indexUrl: dataIndexUrl,
      signal: controller.signal,
      reloadAll: loadRequest.reloadAll,
      cacheGeneration: loadRequest.cacheGeneration,
      onManifest: (manifest) => setDataState((current) => ({ ...current, manifest })),
      onProgress: ({ loaded, total }) => setProgress({
        attempt: loadRequest.attempt,
        loaded,
        total,
      }),
    })
      .then(({ data, sourceData, cacheGeneration }) => {
        if (loadRequest.reloadAll && cacheGeneration) {
          saveCacheGeneration(cacheGeneration);
        }
        setDataState({ data, manifest: data, sourceData, error: null });
      })
      .catch((error) => {
        if (isLoadCancelled(error)) return;
        setDataState((current) => ({ ...current, data: null, sourceData: null, error }));
      });
    return () => controller.abort();
  }, [loadRequest]);

  const toggleMode = () => {
    const next = mode === 'dark' ? 'light' : 'dark';
    setMode(next);
    try { window.localStorage.setItem('rocjitsu-color-mode', next); } catch { /* Theme persistence is optional. */ }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <DashboardShell data={dataState.data ?? emptyDashboardData} state={state} loading={!dataState.data && !dataState.error} header={<DashboardHeader
        tab={state.tab}
        data={dataState.data ?? dataState.manifest}
        dataError={dataState.error}
        downloadData={dataState.sourceData}
        loading={!dataState.data && !dataState.error}
        mode={mode}
        onReloadData={() => {
          setDataState({ data: null, manifest: null, sourceData: null, error: null });
          setLoadRequest((current) => ({
            ...current,
            attempt: current.attempt + 1,
            reloadAll: true,
          }));
        }}
        onToggleMode={toggleMode}
      />}>
      {state.routeError && <Alert data-testid="dashboard-route-error" severity="warning" sx={{ mx: { xs: 2, md: 3 }, mt: 2 }} action={<Button color="inherit" onClick={state.clearRouteError}>Clear invalid selection</Button>}>{state.routeError}</Alert>}
      {!dataState.data && !dataState.error && (
        <LoadingDataState
          progress={
            progress.attempt === loadRequest.attempt
              ? progress
              : { loaded: 0, total: 0 }
          }
        />
      )}
      {dataState.error && (
        <Dashboard
          data={emptyDashboardData}
          state={state}
          dataError={dataState.error}
          onRetry={() => {
            setDataState((current) => ({ ...current, error: null }));
            setLoadRequest((current) => ({
              ...current,
              attempt: current.attempt + 1,
            }));
          }}
        />
      )}
      {dataState.data && <Dashboard data={dataState.data} state={state} />}
      <Box component="footer" sx={{ borderTop: 1, borderColor: 'divider', mx: { xs: 2, md: 3 }, py: 2, color: 'text.secondary' }}>
        <Typography variant="caption" component="p">Copyright © 2025–2026 Advanced Micro Devices, Inc.</Typography>
        <Typography variant="caption" component="p">MIT License</Typography>
      </Box>
      </DashboardShell>
    </ThemeProvider>
  );
}
