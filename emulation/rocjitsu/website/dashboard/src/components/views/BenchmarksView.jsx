import { fontSizes, fontWeights } from '../../theme/typography';
import { useMemo, useState } from 'react';
import { Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, MenuItem, Paper, Stack, Typography } from '@mui/material';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import AddRoundedIcon from '@mui/icons-material/AddRounded';
import { BenchmarkGridPicker } from '../benchmarks/BenchmarkPicker';
import BenchmarkHistoryChart from '../benchmarks/BenchmarkHistoryChart';
import BenchmarkResultDialog from '../benchmarks/BenchmarkResultDialog';
import ChartPointSelector from '../benchmarks/ChartPointSelector';
import SectionCard from '../shared/SectionCard';
import AnchoredSelect from '../shared/AnchoredSelect';
import ActionButton from '../shared/ActionButton';
import { CONTROL_HEIGHT } from '../shared/segmentedControlSx';

import ThreadPolicySwatch from '../shared/ThreadPolicySwatch';
import { targetColor } from '../../theme/targetColors';
import { shapeTokens } from '../../theme/tokens';
import CategoryTag from '../shared/CategoryTag';
import { benchmarkPlottedData, benchmarkResultChoices, boundedGridSelection } from '../benchmarks/benchmarkExplorer';
import { selectBenchmarkCatalog, selectBenchmarkSeries, selectDefaultBenchmarks } from '../../data/selectors';
import { DEFAULT_BENCHMARK_RANGE, historyRanges } from '../../config/historyRanges';
import { MAX_BENCHMARK_CHARTS } from '../../config/benchmarkGrid';
import { formatFullDate } from '../../utils/formatters';

export default function BenchmarksView({ data, filters, historyRange: controlledRange, onRangeChange, initialBenchmarkIds, selectedBenchmarks, onBenchmarksChange, selectedRunIds = [], onSelectRun, onClearSelectedRuns }) {
  const catalog = useMemo(() => selectBenchmarkCatalog(data, filters), [data, filters]);
  const [localRange, setLocalRange] = useState(DEFAULT_BENCHMARK_RANGE);
  const range = controlledRange ?? localRange;
  const changeRange = (next) => { setLocalRange(next); onRangeChange?.(next); };
  const rangedData = useMemo(() => benchmarkPlottedData(data, range), [data, range]);
  const defaultSelection = useMemo(() => selectedBenchmarks === null ? selectDefaultBenchmarks(rangedData, filters) : null, [selectedBenchmarks, rangedData, filters]);
  const [localSelection, setLocalSelection] = useState(() => {
    if (selectedBenchmarks !== undefined) return [];
    return boundedGridSelection(initialBenchmarkIds
      ? initialBenchmarkIds.map((id) => catalog.all.find((test) => test.id === id) ?? { id, name: id, suite: 'Unknown suite' })
      : selectDefaultBenchmarks(rangedData, filters));
  });
  // Undefined preserves the standalone initialBenchmarkIds API. Controlled null
  // derives first-use defaults without writing state; controlled [] stays empty.
  const selection = selectedBenchmarks === undefined ? localSelection : selectedBenchmarks ?? defaultSelection;
  const changeSelection = (tests) => {
    const next = boundedGridSelection(tests);
    if (selectedBenchmarks === undefined) setLocalSelection(next);
    onBenchmarksChange?.(next);
  };
  // Retain the chosen workload identity across scope/data updates, including a
  // workload no longer present in the labenchmark set. Never silently pick another.
  const selected = selection.map((test) => catalog.all.find((current) => current.id === test.id) ?? test);
  const [draft, setDraft] = useState(null);
  const [inspectedBenchmark, setInspectedBenchmark] = useState(null);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [unavailable, setUnavailable] = useState(null);
  const [inspectedChoiceId, setInspectedChoiceId] = useState('');
  const availableIds = new Set(catalog.available.map((test) => test.id));
  const resultChoices = useMemo(() => inspectedBenchmark ? benchmarkResultChoices(selectBenchmarkSeries(rangedData, filters, inspectedBenchmark.id)) : [], [rangedData, filters, inspectedBenchmark]);
  const openRecord = (record) => { onSelectRun?.(record.run.runId); setSelectedRecord(record); };
  const inspectedChoice = resultChoices.find((choice) => choice.id === inspectedChoiceId) ?? resultChoices[0];
  const closeInspector = () => {
    if (selectedRecord) return; // Only the top result dialog handles Escape/backdrop.
    setInspectedBenchmark(null); setInspectedChoiceId(''); setUnavailable(null);
  };
  const activateChoice = (choice) => {
    if (choice.record) openRecord(choice.record);
    else { onSelectRun?.(choice.run.runId); setUnavailable(choice); }
  };
  // Each card gets only the runs the range plots (on 1W, every run of the last 7 commit days), the same set Inspect results
  // and the default selection use, so no card scans the whole history just to cut it down again.
  const chartProps = { data: rangedData, filters, range, selectedRunIds, onSelectRecord: openRecord };
  return (
    <Box data-testid="benchmark-explorer" sx={{ minWidth: 0 }}>
      <SectionCard title="Benchmark history" subtitle="Selected workloads across the configurations checked in the sidebar" sx={{ minWidth: 0 }} action={<ActionButton startIcon={<AddRoundedIcon />} disabled={!catalog.all.length} onClick={() => setDraft(selected)}>Add benchmarks</ActionButton>}>
        <Stack direction="row" sx={{ justifyContent: 'space-between', flexWrap: 'wrap', alignItems: 'center', gap: 1, mb: '12px' }}>
          <Typography variant="caption" color="text.secondary">{filters.targets.length} targets · {filters.suites.length} suites · {(filters.modes ?? []).length} thread policies · develop only</Typography>
          <Stack direction="row" sx={{ alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
            <Typography role="status" variant="caption" color="text.secondary">{selected.length} of {MAX_BENCHMARK_CHARTS} charts{selected.length === MAX_BENCHMARK_CHARTS ? ' · Remove a workload to free a slot' : ''}</Typography>

          </Stack>
        </Stack>
{/* One timeframe for every chart below: a labelled dropdown bar, unlike Overview's per-chart segmented control. */}
        <Stack direction="row" data-testid="benchmark-timeframe" sx={{ alignItems: 'center', flexWrap: 'wrap', gap: '12px', mb: 2, px: 2, py: '10px', border: 1, borderColor: 'dashboard.border', borderRadius: shapeTokens.controlRadius, bgcolor: 'dashboard.subtleSurface' }}>
          <Typography id="benchmark-timeframe-label" variant="body2" sx={{ fontWeight: fontWeights.emphasized }}>Timeframe</Typography>
          <AnchoredSelect size="small" value={range} onChange={(event) => changeRange(event.target.value)} aria-label="Benchmark history timeframe" labelId="benchmark-timeframe-label"
            sx={{ minWidth: 200, height: CONTROL_HEIGHT, bgcolor: 'background.paper', '& .MuiOutlinedInput-notchedOutline legend': { display: 'none' }, '& .MuiOutlinedInput-notchedOutline': { top: 0 } }}>
            {historyRanges.map(({ id, label }) => <MenuItem key={id} value={id}>{label}</MenuItem>)}
          </AnchoredSelect>
          <Typography variant="caption" color="text.secondary">Applies to all {selected.length} charts below · by commit date, UTC</Typography>
        </Stack>
        <Stack direction="row" aria-label="Chart category and thread policy legend" sx={{ flexWrap: 'wrap', alignItems: 'center', gap: 2, borderTop: 1, borderColor: 'divider', pt: '12px', mb: 2, color: 'text.secondary' }}>
          {(filters.modes ?? []).flatMap((mode) => filters.targets.map((target) => <Typography key={`${target}/${mode}`} variant="caption" sx={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}><ThreadPolicySwatch policy={mode} color={(theme) => targetColor(theme, target)} legend />{target} {mode}</Typography>))}
          <Typography variant="caption" color="text.secondary" sx={{ ml: { sm: 'auto' } }}>Wall-clock seconds · measured results</Typography>
        </Stack>
        {!catalog.all.length && <Typography role="status" color="text.secondary" sx={{ py: 3 }}>No benchmarks available in the published catalog.</Typography>}
        {catalog.all.length > 0 && !catalog.available.length && <Typography role="status" color="text.secondary" sx={{ py: 3 }}>No benchmarks in the selected targets, suites and thread policies. Retained workloads are shown as unavailable.</Typography>}
        <Box data-testid="benchmark-grid" sx={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', '@media (min-width:600px) and (max-width:899px)': { gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }, '@media (min-width:900px)': { gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }, '@media (min-width:1200px)': { gridTemplateColumns: 'repeat(3, minmax(0, 1fr))' }, gap: 2 }}>
          {selected.map((benchmark) => <Paper data-testid="benchmark-grid-card" key={benchmark.id} variant="outlined" sx={{ p: shapeTokens.cardPadding, minWidth: 0 }}>
            <Stack direction="row" sx={{ alignItems: 'flex-start', justifyContent: 'space-between', gap: 1, mb: 1 }}>
              <Box sx={{ minWidth: 0 }}><Typography variant="h3" sx={{ fontSize: fontSizes.body, lineHeight: '20px', fontWeight: fontWeights.emphasized, overflowWrap: 'anywhere' }}>{benchmark.name}</Typography><Typography variant="caption" component="div" color="text.secondary" sx={{ overflowWrap: 'anywhere' }}>{benchmark.id}</Typography><CategoryTag label={benchmark.suite} sx={{ mt: '6px' }} /></Box>
              <IconButton size="small" aria-label={`Remove ${benchmark.name} from grid`} onClick={() => changeSelection(selected.filter((test) => test.id !== benchmark.id))} sx={{ width: { xs: 44, sm: 32 }, height: { xs: 44, sm: 32 } }}><CloseRoundedIcon sx={{ fontSize: 18 }} /></IconButton>
            </Stack>
            {availableIds.has(benchmark.id) ? <BenchmarkHistoryChart key={`${benchmark.id}:${range}`} {...chartProps} benchmark={benchmark} height={230} /> : <Box role="status" sx={{ minHeight: 230, display: 'grid', placeItems: 'center', textAlign: 'center', color: 'text.secondary', fontSize: fontSizes.body }}>Not in this configuration</Box>}
            <Stack direction="row" sx={{ flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '4px', mt: '14px' }}>
              <Typography variant="caption" color="text.muted" sx={{ fontSize: fontSizes.caption }}>Published · {data.generatedAt ? formatFullDate(data.generatedAt) : 'Not provided'}</Typography>
              <ActionButton size="small" aria-label={`Inspect ${benchmark.name} results`} onClick={() => { setUnavailable(null); setInspectedBenchmark(benchmark); }}>Inspect results</ActionButton>
            </Stack>
          </Paper>)}
        </Box>
        {!selected.length && catalog.all.length > 0 && <Typography role="status" color="text.secondary" sx={{ py: 3 }}>No benchmark graphs selected. Add benchmarks to begin.</Typography>}
        <Stack direction="row" sx={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mt: '12px' }}>
          <Typography variant="caption" color="text.muted">Each chart is one workload. Hover a point to see the run behind it.</Typography>
          {selectedRunIds.length > 0 && <ActionButton size="small" onClick={onClearSelectedRuns}>Clear selected runs ({selectedRunIds.length})</ActionButton>}
        </Stack>
      </SectionCard>
      <Dialog open={draft !== null} onClose={() => setDraft(null)} fullWidth maxWidth="sm" aria-labelledby="add-benchmarks-title">
        <DialogTitle id="add-benchmarks-title">Add benchmarks</DialogTitle>
        <DialogContent><Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>{`Choose up to ${MAX_BENCHMARK_CHARTS} workloads. Options follow the selected targets, suites and thread policies.`}</Typography><BenchmarkGridPicker availableOptions={catalog.available} selected={draft ?? []} onChange={setDraft} /></DialogContent>
        <DialogActions><Button color="inherit" onClick={() => setDraft(null)}>Cancel</Button><ActionButton onClick={() => { changeSelection(draft ?? []); setDraft(null); }}>Apply selection</ActionButton></DialogActions>
      </Dialog>
      <Dialog open={Boolean(inspectedBenchmark)} disableEnforceFocus={Boolean(selectedRecord)} onClose={closeInspector} fullWidth maxWidth="sm" aria-labelledby="inspect-results-title">
        <DialogTitle id="inspect-results-title">Inspect results · {inspectedBenchmark?.name}</DialogTitle>
        <DialogContent>
          {!resultChoices.length ? <Typography color="text.secondary">No published results in this selected configuration and period.</Typography> : <ChartPointSelector label={`${inspectedBenchmark?.name} result`} description="Search and select a published attempt, including failed, timed-out and unavailable results." descriptionId="inspect-result-help" options={resultChoices} selectedId={inspectedChoiceId} onSelectionChange={(id) => { setInspectedChoiceId(id); setUnavailable(null); }} />}
          {unavailable && <Alert severity="info" sx={{ mt: 1 }}>Unavailable · {unavailable.target} · {unavailable.mode} · {unavailable.run.runId}. No published result; this is not a failed or zero-duration measurement.</Alert>}
        </DialogContent>
        <DialogActions sx={{ px: 3, py: '12px', flexWrap: 'wrap', gap: 1 }}><Button color="inherit" onClick={closeInspector}>Close</Button><ActionButton disabled={!inspectedChoice} onClick={() => activateChoice(inspectedChoice)}>Open result details</ActionButton></DialogActions>
      </Dialog>
      <BenchmarkResultDialog record={selectedRecord} repository={data.repository} onClose={() => setSelectedRecord(null)} />
    </Box>
  );
}
