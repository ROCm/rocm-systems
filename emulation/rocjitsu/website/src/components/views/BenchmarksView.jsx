import { useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  IconButton,
  Paper,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import ClearAllRoundedIcon from '@mui/icons-material/ClearAllRounded';
import MouseRoundedIcon from '@mui/icons-material/MouseRounded';
import TouchAppRoundedIcon from '@mui/icons-material/TouchAppRounded';
import BenchmarkPicker, { BenchmarkGridPicker } from '../benchmarks/BenchmarkPicker';
import BenchmarkHistoryChart from '../benchmarks/BenchmarkHistoryChart';
import BenchmarkResultDialog from '../benchmarks/BenchmarkResultDialog';
import RunDetailsDialog from '../benchmarks/RunDetailsDialog';
import HistoricalRecords from '../benchmarks/HistoricalRecords';
import AggregatePerformanceChart from '../benchmarks/AggregatePerformanceChart';
import SectionCard from '../shared/SectionCard';
import { selectBenchmarkCatalog } from '../../data/selectors';

const MAX_GRID_BENCHMARKS = 8;

function ExplorerToggleButton({ enabled, label, ariaFeature, icon, onClick }) {
  return (
    <Button
      size="small"
      variant="outlined"
      color={enabled ? 'primary' : 'inherit'}
      startIcon={icon}
      aria-label={`${enabled ? 'Disable' : 'Enable'} ${ariaFeature}`}
      aria-pressed={enabled}
      onClick={onClick}
      sx={{
        minHeight: 40,
        bgcolor: (theme) => enabled ? alpha(theme.palette.primary.main, 0.06) : 'transparent',
        whiteSpace: 'nowrap',
      }}
    >
      {label} · {enabled ? 'On' : 'Off'}
    </Button>
  );
}

function FilterOverrideAlert({ benchmarks, available, onReturn }) {
  if (benchmarks.length === 0) return null;
  return (
    <Alert
      severity="info"
      variant="outlined"
      action={available.length > 0 ? <Button color="inherit" size="small" onClick={onReturn}>Return to filtered benchmarks</Button> : null}
      sx={{ mb: 2 }}
    >
      <Typography variant="body2" fontWeight={700}>Outside global Suite filter</Typography>
      <Typography variant="caption">
        {benchmarks.length === 1
          ? 'This benchmark belongs to a suite excluded by the global filter. This local override applies only to Benchmark Explorer.'
          : 'Some selected benchmarks belong to suites excluded by the global filter. This local override applies only to Benchmark Explorer.'}
      </Typography>
    </Alert>
  );
}

export default function BenchmarksView({
  data,
  filters,
  selectedRunIds,
  onSelectRun,
  onClearSelectedRuns,
}) {
  const catalog = useMemo(() => selectBenchmarkCatalog(data, filters), [data, filters]);
  const initialId = catalog.available[0]?.id ?? catalog.all[0]?.id ?? '';
  const [mode, setMode] = useState(selectedRunIds.length > 0 ? 'aggregate' : 'single');
  const [selectedId, setSelectedId] = useState(initialId);
  const [gridIds, setGridIds] = useState(initialId ? [initialId] : []);
  const [showAll, setShowAll] = useState(false);
  const [showDetailsOnClick, setShowDetailsOnClick] = useState(true);
  const [scrollZoomEnabled, setScrollZoomEnabled] = useState(true);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [selectedRunDetails, setSelectedRunDetails] = useState(null);
  const selectedTest = catalog.all.find((test) => test.id === selectedId) ?? catalog.all[0];
  const selectedGridTests = gridIds.map((id) => catalog.all.find((test) => test.id === id)).filter(Boolean);
  const availableIds = new Set(catalog.available.map((test) => test.id));
  const outsideSingleFilter = selectedTest && !availableIds.has(selectedTest.id) ? [selectedTest] : [];
  const outsideGridFilter = selectedGridTests.filter((test) => !availableIds.has(test.id));
  const selectGraphPoint = (record) => {
    onSelectRun(record.run.runId);
    if (showDetailsOnClick) setSelectedRecord(record);
  };
  const selectAggregatePoint = (run) => {
    onSelectRun(run.runId);
    if (showDetailsOnClick) setSelectedRunDetails(run);
  };
  // Explicit "open" actions always show details; the click toggle governs chart clicks only.
  const openGraphPoint = (record) => {
    onSelectRun(record.run.runId);
    setSelectedRecord(record);
  };
  const openAggregatePoint = (run) => {
    onSelectRun(run.runId);
    setSelectedRunDetails(run);
  };

  const returnSingleToFilters = () => {
    if (catalog.available.length === 0) return;
    setSelectedId(catalog.available[0].id);
    setShowAll(false);
  };
  const returnGridToFilters = () => {
    const next = selectedGridTests.filter((test) => availableIds.has(test.id));
    setGridIds((next.length > 0 ? next : catalog.available.slice(0, 1)).map((test) => test.id));
    setShowAll(false);
  };

  if (!selectedTest) {
    return <Paper variant="outlined" sx={{ p: 4 }}><Typography color="text.secondary">No benchmarks are available for the selected suites.</Typography></Paper>;
  }

  const interactionControls = (
    <Stack direction="row" sx={{ alignItems: 'center', justifyContent: { xs: 'flex-start', md: 'flex-end' }, flexWrap: 'wrap', gap: 0.75 }}>
      <ExplorerToggleButton
        enabled={showDetailsOnClick}
        label="Click details"
        ariaFeature="details on click"
        icon={<TouchAppRoundedIcon />}
        onClick={() => setShowDetailsOnClick((current) => !current)}
      />
      <ExplorerToggleButton
        enabled={scrollZoomEnabled}
        label="Scroll zoom"
        ariaFeature="scroll zoom"
        icon={<MouseRoundedIcon />}
        onClick={() => setScrollZoomEnabled((current) => !current)}
      />
    </Stack>
  );

  return (
    <Box sx={{ display: 'grid', gap: 1.75 }}>
      <SectionCard
        title="Benchmark Explorer"
        subtitle={mode === 'single'
          ? 'One benchmark across all official attempts, including reruns'
          : mode === 'grid'
            ? `Up to ${MAX_GRID_BENCHMARKS} benchmark histories across all official attempts`
            : 'Selected-suite duration by target across all official attempts, including reruns'}
        action={(
          <Stack sx={{ alignItems: { xs: 'flex-start', sm: 'flex-end' }, gap: 0.75 }}>
            <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'flex-end', flexWrap: 'wrap', gap: 0.75 }}>
              <Button
                size="small"
                color="inherit"
                startIcon={<ClearAllRoundedIcon />}
                disabled={selectedRunIds.length === 0}
                onClick={onClearSelectedRuns}
              >
                Clear selected runs ({selectedRunIds.length})
              </Button>
              <ToggleButtonGroup
                exclusive
                size="small"
                value={mode}
                onChange={(_, nextMode) => {
                  if (!nextMode) return;
                  if (nextMode === 'grid' && gridIds.length === 0) setGridIds([selectedTest.id]);
                  setMode(nextMode);
                }}
                aria-label="Benchmark display mode"
              >
                <ToggleButton value="single">Single</ToggleButton>
                <ToggleButton value="grid">Grid</ToggleButton>
                <ToggleButton value="aggregate">Aggregate</ToggleButton>
              </ToggleButtonGroup>
            </Stack>
          </Stack>
        )}
      >
        {mode === 'aggregate' ? (
          <>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: { xs: 'flex-start', md: 'flex-end' }, gap: 0.75, mb: 1.5 }}>
              <Typography variant="caption" color="text.secondary">
                {data.runs.length} official attempts · Selected runs remain visible while zooming · {scrollZoomEnabled
                  ? 'Scroll, pinch, or use the slider to change the visible range'
                  : 'Use the slider to change the visible range'}
              </Typography>
              {interactionControls}
            </Box>
            <AggregatePerformanceChart
              data={data}
              filters={filters}
              selectedRunIds={selectedRunIds}
              onSelectRun={selectAggregatePoint}
              onOpenRun={openAggregatePoint}
              showDetailsOnClick={showDetailsOnClick}
              scrollZoomEnabled={scrollZoomEnabled}
            />
          </>
        ) : mode === 'single' ? (
          <>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'minmax(320px, 720px) auto' }, alignItems: 'start', gap: 1.5, mb: 2 }}>
              <BenchmarkPicker
                allOptions={catalog.all}
                availableOptions={catalog.available}
                hiddenCount={catalog.hiddenCount}
                selected={selectedTest}
                showingAll={showAll}
                onChange={(test) => {
                  setSelectedId(test.id);
                }}
                onToggleScope={() => setShowAll((current) => !current)}
              />
              {interactionControls}
            </Box>
            <FilterOverrideAlert benchmarks={outsideSingleFilter} available={catalog.available} onReturn={returnSingleToFilters} />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.75 }}>
              {showDetailsOnClick
                ? 'Dotted segments bridge unavailable measurements. Click any result marker to select its run and open pass/failure details.'
                : 'Dotted segments bridge unavailable measurements. Click any result marker to select or clear its run.'}
            </Typography>
            <BenchmarkHistoryChart
              data={data}
              filters={filters}
              benchmark={selectedTest}
              selectedRunIds={selectedRunIds}
              onSelectRecord={selectGraphPoint}
              onOpenRecord={openGraphPoint}
              onSelectRun={onSelectRun}
              showDetailsOnClick={showDetailsOnClick}
              showPointSelector
              scrollZoomEnabled={scrollZoomEnabled}
            />
          </>
        ) : (
          <>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'minmax(360px, 720px) auto' }, alignItems: 'start', gap: 1.5, mb: 2 }}>
              <BenchmarkGridPicker
                allOptions={catalog.all}
                availableOptions={catalog.available}
                hiddenCount={catalog.hiddenCount}
                selected={selectedGridTests}
                showingAll={showAll}
                maxSelected={MAX_GRID_BENCHMARKS}
                onChange={(tests) => {
                  setGridIds(tests.map((test) => test.id));
                }}
                onToggleScope={() => setShowAll((current) => !current)}
              />
              {interactionControls}
            </Box>
            <FilterOverrideAlert benchmarks={outsideGridFilter} available={catalog.available} onReturn={returnGridToFilters} />
            {selectedGridTests.length > 0 ? (
              <Box data-testid="benchmark-grid" sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0, 1fr)', lg: 'repeat(2, minmax(0, 1fr))' }, gap: 1.5 }}>
                {selectedGridTests.map((benchmark) => (
                  <Paper key={benchmark.id} variant="outlined" sx={{ minWidth: 0, p: { xs: 1.25, sm: 1.75 }, borderRadius: 2.5 }}>
                    <Stack direction="row" sx={{ alignItems: 'flex-start', justifyContent: 'space-between', gap: 1, mb: 0.5 }}>
                      <Box sx={{ minWidth: 0 }}>
                        <Typography variant="h3">{benchmark.name}</Typography>
                        <Chip label={benchmark.suite} size="small" variant="outlined" sx={{ mt: 0.65 }} />
                      </Box>
                      <Tooltip title="Hide graph">
                        <IconButton
                          size="small"
                          aria-label={`Hide ${benchmark.name} graph`}
                          onClick={() => {
                            setGridIds((current) => current.filter((id) => id !== benchmark.id));
                          }}
                        >
                          <CloseRoundedIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Stack>
                    <BenchmarkHistoryChart
                      data={data}
                      filters={filters}
                      benchmark={benchmark}
                      height={290}
                      selectedRunIds={selectedRunIds}
                      onSelectRecord={selectGraphPoint}
                      onOpenRecord={openGraphPoint}
                      onSelectRun={onSelectRun}
                      showDetailsOnClick={showDetailsOnClick}
                      showPointSelector
                      scrollZoomEnabled={scrollZoomEnabled}
                    />
                  </Paper>
                ))}
              </Box>
            ) : (
              <Paper variant="outlined" sx={{ p: 5, borderStyle: 'dashed', textAlign: 'center' }}>
                <Typography fontWeight={700}>No benchmark graphs selected</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>Use the checkbox list above to display up to eight graphs.</Typography>
              </Paper>
            )}
          </>
        )}
      </SectionCard>

      {mode === 'single' && (
        <HistoricalRecords
          key={`${selectedTest.id}:${filters.targets.join(',')}:${filters.suites.join(',')}`}
          data={data}
          filters={filters}
          benchmark={selectedTest}
          onSelectRecord={openGraphPoint}
        />
      )}

      <BenchmarkResultDialog
        record={selectedRecord}
        repository={data.repository}
        onClose={() => setSelectedRecord(null)}
      />
      <RunDetailsDialog
        run={selectedRunDetails}
        filters={filters}
        repository={data.repository}
        onClose={() => setSelectedRunDetails(null)}
      />
    </Box>
  );
}
