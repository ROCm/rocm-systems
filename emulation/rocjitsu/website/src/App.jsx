import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Badge,
  Box,
  Button,
  Chip,
  Container,
  CssBaseline,
  LinearProgress,
  Paper,
  Skeleton,
  Stack,
  Tab,
  Tabs,
  ThemeProvider,
  Typography,
} from '@mui/material';
import AccessTimeRoundedIcon from '@mui/icons-material/AccessTimeRounded';
import DashboardHeader from './components/layout/DashboardHeader';
import FiltersBar from './components/layout/FiltersBar';
import OverviewView from './components/overview/OverviewView';
import BenchmarksView from './components/views/BenchmarksView';
import CompareRunsView from './components/views/CompareRunsView';
import FailuresView from './components/views/FailuresView';
import PluginComparisonView from './components/views/PluginComparisonView';
import { isLoadCancelled, loadDashboardDataFiles } from './data/dashboardData';
import { selectFailures, selectOverview } from './data/selectors';
import { useDashboardState } from './hooks/useDashboardState';
import { visuallyHiddenStyles } from './theme/styles';
import { createDashboardTheme } from './theme/theme';
import { formatFullDate, shortSha } from './utils/formatters';

const dataMetadataUrl = new URL(`${import.meta.env.BASE_URL}data/metadata.json`, document.baseURI).href;
const dataIndexUrl = new URL(`${import.meta.env.BASE_URL}data/index.json`, document.baseURI).href;

function LoadingDataState({ progress }) {
  const determinate = progress.total > 0;
  return (
    <Box component="main" sx={{ minHeight: 'calc(100vh - 76px)' }}>
      <Container maxWidth={false} sx={{ maxWidth: 1600, px: { xs: 2, sm: 3, xl: 4 }, pt: { xs: 2.5, md: 3.5 }, pb: 6 }}>
        <DashboardHero />
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
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.4 }}>
              The dashboard shell is ready while run files are fetched and validated.
            </Typography>
            {determinate && (
              <Typography data-testid="dashboard-load-progress" variant="body2" color="text.secondary" sx={{ mt: 0.4 }}>
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
  repository: null,
  generatedAt: null,
  isBeta: false,
  runs: [],
  pluginRuns: [],
  testCatalog: [],
  testCatalogs: [],
  targets: [],
  suites: [],
  plugins: [],
  latestRun: null,
  latestCommitRun: null,
  backfillRunIds: new Set(),
};

function emptyOverview(range) {
  return {
    candidate: null,
    baseline: null,
    latestTests: [],
    comparisons: [],
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
      runCount: 0,
      currentDuration: null,
      firstRun: null,
      latestRun: null,
      durationDelta: null,
      comparisonLabel: '—',
      summary: '—',
      description: 'No benchmark history is available',
      series: [],
    },
  };
}

function EmptyDataState({ error, onRetry }) {
  return <Dashboard data={emptyDashboardData} dataError={error} onRetry={onRetry} />;
}

function DashboardHero({ data = null }) {
  return (
    <Stack direction={{ xs: 'column', md: 'row' }} sx={{ justifyContent: 'space-between', alignItems: { xs: 'flex-start', md: 'flex-end' }, gap: 2, mb: 2.5 }}>
      <Box>
        <Typography component="h1" variant="h1">RocJitsu Performance Health</Typography>
        <Typography color="text.secondary" sx={{ mt: 0.7, maxWidth: 760 }}>
          Track benchmark performance, regressions, and run coverage across selected GFX targets.
        </Typography>
      </Box>
      <Paper data-testid="latest-commit-run" variant="outlined" sx={{ minWidth: 245, py: 1.1, px: 1.5, borderRadius: 2.5, bgcolor: 'action.hover' }}>
        <Stack direction="row" sx={{ alignItems: 'center', gap: 1 }}>
          <AccessTimeRoundedIcon color="primary" sx={{ fontSize: 18 }} />
          <Box sx={{ flex: 1 }}>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>Latest commit run</Typography>
            {data?.latestCommitRun ? (
              <Stack direction="row" sx={{ gap: 0.8, alignItems: 'center' }}>
                <Typography variant="caption" fontWeight={700}>{formatFullDate(data.latestCommitRun.timestamp)}</Typography>
                <Chip label={shortSha(data.latestCommitRun)} size="small" sx={{ height: 20, fontFamily: 'monospace', fontSize: 10 }} />
              </Stack>
            ) : data ? (
              <Stack direction="row" sx={{ gap: 0.8, alignItems: 'center' }}>
                <Typography variant="caption" color="text.secondary" fontWeight={700}>—</Typography>
                <Chip label="—" size="small" disabled sx={{ height: 20, fontFamily: 'monospace', fontSize: 10 }} />
              </Stack>
            ) : (
              <Skeleton variant="text" animation="wave" width="88%" sx={{ fontSize: 16 }} />
            )}
          </Box>
        </Stack>
      </Paper>
    </Stack>
  );
}

function Dashboard({ data, dataWarnings = [], dataError = null, onRetry = null }) {
  const state = useDashboardState(data);
  const hasData = data.runs.length > 0;
  // Overview derives the whole history, so it stays uncomputed while another tab owns the view.
  const overview = useMemo(
    () => (state.tab === 'overview'
      ? hasData
        ? selectOverview(data, state.filters, state.historyRange)
        : emptyOverview(state.historyRange)
      : null),
    [data, hasData, state.filters, state.historyRange, state.tab],
  );
  const failureCount = useMemo(() => selectFailures(data, state.filters).length, [data, state.filters]);
  const openRunComparison = (runIds) => {
    const selectedRuns = runIds
      .map((runId) => data.runs.find((run) => run.runId === runId))
      .filter(Boolean);
    if (selectedRuns.length !== 2) return;
    state.setComparisonCandidateId(selectedRuns[0].runId);
    state.setComparisonBaselineId(selectedRuns[1].runId);
    state.setTab('compare');
  };
  const openRunInExplorer = (runId) => {
    state.setExplorerRunIds([runId]);
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
        <Container maxWidth={false} sx={{ maxWidth: 1600, px: { xs: 2, sm: 3, xl: 4 }, pt: { xs: 2.5, md: 3.5 }, pb: 6 }}>
          <DashboardHero data={data} />

          {dataError && (
            <Alert
              data-testid="dashboard-data-error"
              severity="warning"
              variant="outlined"
              action={<Button color="inherit" size="small" onClick={onRetry}>Retry</Button>}
              sx={{ mb: 1.75 }}
            >
              <Typography fontWeight={700}>No available test data</Typography>
              {dataError.message !== 'No available test data' && (
                <Typography variant="body2" sx={{ mt: 0.5 }}>{dataError.message}</Typography>
              )}
              <Typography variant="body2" sx={{ mt: 1 }}>Dashboard values will remain empty until benchmark data is published to the site.</Typography>
            </Alert>
          )}

          {dataWarnings.length > 0 && (
            <Alert data-testid="invalid-run-warning" severity="warning" variant="outlined" sx={{ mb: 1.75 }}>
              <Typography fontWeight={700}>
                Skipped {dataWarnings.length} invalid run {dataWarnings.length === 1 ? 'file' : 'files'}
              </Typography>
              {dataWarnings.slice(0, 3).map((warning) => (
                <Typography key={`${warning.runFile}:${warning.message}`} component="div" variant="caption" sx={{ mt: 0.35, overflowWrap: 'anywhere' }}>
                  <code>{warning.runFile}</code> — {warning.message}
                </Typography>
              ))}
              {dataWarnings.length > 3 && (
                <Typography variant="caption" color="text.secondary">And {dataWarnings.length - 3} more.</Typography>
              )}
            </Alert>
          )}

          <FiltersBar data={data} state={state} disabled={!hasData} />

          <Paper data-testid="dashboard-navigation" variant="outlined" sx={{ mt: 1.75, mb: 1.75, borderRadius: 3, overflow: 'hidden' }}>
            <Tabs
              value={state.tab}
              onChange={(_, value) => {
                if (value === 'benchmarks') state.setExplorerRunIds([]);
                state.setTab(value);
              }}
              variant="scrollable"
              scrollButtons="auto"
              aria-label="Dashboard views"
              sx={{ px: { xs: 0.5, sm: 1.25 }, minHeight: 50 }}
            >
              <Tab value="overview" label="Overview" />
              <Tab value="benchmarks" label="Benchmarks" />
              <Tab value="compare" label="Run Comparison" />
              <Tab value="plugins" label="Plugin Comparison" />
              <Tab
                value="failures"
                sx={{ minWidth: 112, px: 2.75 }}
                label={<Badge badgeContent={failureCount} color="error" max={99} sx={{ '& .MuiBadge-badge': { right: -13, top: 7 } }}>Failures</Badge>}
              />
            </Tabs>
          </Paper>

          {state.tab === 'overview' && (
            <OverviewView
              viewModel={overview}
              data={data}
              state={state}
              onCompareRun={openRunComparison}
              onExploreRun={openRunInExplorer}
            />
          )}
          {state.tab === 'benchmarks' && (
            <BenchmarksView
              data={data}
              filters={state.filters}
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
            />
          )}
          {state.tab === 'plugins' && <PluginComparisonView data={data} filters={state.filters} />}
          {state.tab === 'failures' && <FailuresView data={data} filters={state.filters} />}
        </Container>
      </Box>
      <Box component="footer" sx={{ borderTop: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
        <Container maxWidth={false} sx={{ maxWidth: 1600, px: { xs: 2, sm: 3, xl: 4 }, py: 2.25 }}>
          <Stack direction="row" sx={{ justifyContent: 'flex-end' }}>
            <Typography variant="caption" color="text.secondary">Data schema v{data.schemaVersion ?? '—'}</Typography>
          </Stack>
        </Container>
      </Box>
    </>
  );
}

export default function App() {
  const preferredMode = window.localStorage.getItem('rocjitsu-color-mode')
    ?? (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const [mode, setMode] = useState(preferredMode);
  const [dataState, setDataState] = useState({ data: null, manifest: null, sourceData: null, warnings: [], error: null });
  // Progress is tagged with the attempt that produced it so a retry starts from zero without an
  // extra state reset.
  const [progress, setProgress] = useState({ attempt: 0, loaded: 0, total: 0 });
  const [attempt, setAttempt] = useState(0);
  const theme = useMemo(() => createDashboardTheme(mode), [mode]);

  useEffect(() => {
    const controller = new AbortController();
    loadDashboardDataFiles({
      metadataUrl: dataMetadataUrl,
      indexUrl: dataIndexUrl,
      signal: controller.signal,
      onManifest: (manifest) => setDataState((current) => ({ ...current, manifest })),
      onProgress: ({ loaded, total }) => setProgress({ attempt, loaded, total }),
    })
      .then(({ data, sourceData, warnings }) => {
        setDataState({ data, manifest: data, sourceData, warnings, error: null });
      })
      .catch((error) => {
        if (isLoadCancelled(error)) return;
        setDataState((current) => ({ ...current, data: null, sourceData: null, warnings: [], error }));
      });
    return () => controller.abort();
  }, [attempt]);

  const toggleMode = () => {
    setMode((current) => {
      const next = current === 'dark' ? 'light' : 'dark';
      window.localStorage.setItem('rocjitsu-color-mode', next);
      return next;
    });
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <DashboardHeader
        data={dataState.data ?? dataState.manifest}
        downloadData={dataState.sourceData}
        loading={!dataState.data && !dataState.error}
        mode={mode}
        onToggleMode={toggleMode}
      />
      {!dataState.data && !dataState.error && (
        <LoadingDataState progress={progress.attempt === attempt ? progress : { loaded: 0, total: 0 }} />
      )}
      {dataState.error && (
        <EmptyDataState
          error={dataState.error}
          onRetry={() => {
            setDataState((current) => ({ ...current, error: null }));
            setAttempt((current) => current + 1);
          }}
        />
      )}
      {dataState.data && <Dashboard data={dataState.data} dataWarnings={dataState.warnings} />}
    </ThemeProvider>
  );
}
