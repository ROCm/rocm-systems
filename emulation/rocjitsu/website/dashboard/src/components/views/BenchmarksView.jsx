import { useMemo, useState } from 'react';
import { Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, Paper, Stack, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import AddRoundedIcon from '@mui/icons-material/AddRounded';
import { BenchmarkGridPicker } from '../benchmarks/BenchmarkPicker';
import BenchmarkHistoryChart from '../benchmarks/BenchmarkHistoryChart';
import BenchmarkResultDialog from '../benchmarks/BenchmarkResultDialog';
import ChartPointSelector from '../benchmarks/ChartPointSelector';
import SectionCard from '../shared/SectionCard';
import CategoryTag from '../shared/CategoryTag';
import { benchmarkRangeData, benchmarkResultChoices, boundedGridSelection, explorerCatalog } from '../benchmarks/benchmarkExplorer';
import { selectBenchmarkSeries } from '../../data/selectors';
import { historyRanges } from '../../config/historyRanges';
import { formatFullDate } from '../../utils/formatters';

export default function BenchmarksView({ data, filters, historyRange: controlledRange, onRangeChange, initialBenchmarkIds, selectedBenchmarks, onBenchmarksChange, selectedRunIds = [], onSelectRun, onClearSelectedRuns }) {
  const catalog = useMemo(() => explorerCatalog(data, filters), [data, filters]);
  const [localRange, setLocalRange] = useState('ALL');
  const range = controlledRange ?? localRange;
  const changeRange = (next) => { setLocalRange(next); onRangeChange?.(next); };
  const rangedData = useMemo(() => benchmarkRangeData(data, range), [data, range]);
  const [localSelection, setLocalSelection] = useState(() => boundedGridSelection(initialBenchmarkIds ? initialBenchmarkIds.map((id) => catalog.all.find((test) => test.id === id) ?? { id, name: id, suite: 'Unknown suite' }) : catalog.available.slice(0, 4)));
  // Undefined preserves the standalone initialBenchmarkIds API. Controlled null
  // derives first-use defaults without writing state; controlled [] stays empty.
  const selection = selectedBenchmarks === undefined ? localSelection : selectedBenchmarks ?? catalog.available.slice(0, 4);
  const changeSelection = (tests) => {
    const next = boundedGridSelection(tests);
    if (selectedBenchmarks === undefined) setLocalSelection(next);
    onBenchmarksChange?.(next);
  };
  // Retain the chosen workload identity across scope/data updates, including a
  // workload no longer present in the latest catalog. Never silently pick another.
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
  const chartProps = { data, filters, range, selectedRunIds, onSelectRecord: openRecord, onOpenRecord: openRecord, onSelectRun, showDetailsOnClick: true, showPointSelector: false, scrollZoomEnabled: false };
  return (
    <Box data-testid="benchmark-explorer" sx={{ minWidth: 0 }}>
      <SectionCard title="Benchmark history" subtitle="Selected workloads across the configurations checked in the sidebar" sx={{ minWidth: 0 }} action={<Button variant="outlined" size="small" startIcon={<AddRoundedIcon />} disabled={!catalog.all.length} onClick={() => setDraft(selected)}>Add benchmarks</Button>}>
        <Stack direction="row" sx={{ justifyContent: 'space-between', flexWrap: 'wrap', alignItems: 'center', gap: 1, mb: 1.5 }}>
          <Typography variant="caption" color="text.secondary">{filters.targets.length} targets · {filters.suites.length} suites · {(filters.modes ?? []).length} execution modes · develop only</Typography>
          <Stack direction="row" sx={{ alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
            <Typography role="status" variant="caption" color="text.secondary">{selected.length} of 8 charts{selected.length === 8 ? ' · Remove a workload to free a slot' : ''}</Typography>
            <ToggleButtonGroup exclusive size="small" value={range} onChange={(_, next) => next && changeRange(next)} aria-label="Benchmark history timeframe">
              {historyRanges.map(({ id: value, label }) => <ToggleButton key={value} value={value} aria-label={label} sx={{ minHeight: { xs: 44, sm: 32 }, fontSize: 11, px: 1 }}>{value}</ToggleButton>)}
            </ToggleButtonGroup>
          </Stack>
        </Stack>
        <Stack direction="row" aria-label="Chart category and mode legend" sx={{ flexWrap: 'wrap', alignItems: 'center', gap: 1.25, borderTop: 1, borderColor: 'divider', pt: 1.5, mb: 2 }}>
          {filters.targets.map((target) => <CategoryTag key={target} kind="target" label={target} />)}
          <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}><Box component="span" aria-hidden="true" sx={{ width: 24, borderTop: '2px solid' }} />ST · solid</Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}><Box component="span" aria-hidden="true" sx={{ width: 24, borderTop: '2px dashed' }} />MT · dashed</Typography>
          <Typography variant="caption" color="text.secondary" sx={{ ml: { sm: 'auto' } }}>Wall-clock seconds · measured results</Typography>
        </Stack>
        {!catalog.all.length && <Alert severity="info">No benchmarks available in the published catalog.</Alert>}
        {catalog.all.length > 0 && !catalog.available.length && <Alert severity="info" sx={{ mb: 1.5 }}>No benchmarks in the selected targets, suites and execution modes. Retained workloads are shown as unavailable.</Alert>}
        <Box data-testid="benchmark-grid" sx={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', '@media (min-width:600px) and (max-width:899px)': { gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }, '@media (min-width:1200px)': { gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }, '@media (min-width:1800px)': { gridTemplateColumns: 'repeat(3, minmax(0, 1fr))' }, gap: 2 }}>
          {selected.map((benchmark) => <Paper data-testid="benchmark-grid-card" key={benchmark.id} variant="outlined" sx={{ p: 1.75, minWidth: 0, borderRadius: '8px' }}>
            <Stack direction="row" sx={{ alignItems: 'flex-start', justifyContent: 'space-between', gap: 1, mb: 1 }}>
              <Box sx={{ minWidth: 0 }}><Typography variant="h3" sx={{ fontSize: 14, lineHeight: '20px', fontWeight: 600, overflowWrap: 'anywhere' }}>{benchmark.name}</Typography><CategoryTag label={benchmark.suite} sx={{ mt: 0.75 }} /></Box>
              <IconButton size="small" aria-label={`Remove ${benchmark.name} from grid`} onClick={() => changeSelection(selected.filter((test) => test.id !== benchmark.id))} sx={{ width: { xs: 44, sm: 32 }, height: { xs: 44, sm: 32 } }}><CloseRoundedIcon sx={{ fontSize: 18 }} /></IconButton>
            </Stack>
            {availableIds.has(benchmark.id) ? <BenchmarkHistoryChart key={`${benchmark.id}:${range}`} {...chartProps} benchmark={benchmark} height={230} /> : <Box role="status" sx={{ minHeight: 230, display: 'grid', placeItems: 'center', textAlign: 'center', color: 'text.secondary', fontSize: 14 }}>Not in this configuration</Box>}
            <Stack direction="row" sx={{ flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 0.5 }}>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11 }}>Published · {data.generatedAt ? formatFullDate(data.generatedAt) : 'Not provided'}</Typography>
              <Button size="small" aria-label={`Inspect ${benchmark.name} results`} onClick={() => { setUnavailable(null); setInspectedBenchmark(benchmark); }}>Inspect results</Button>
            </Stack>
          </Paper>)}
        </Box>
        {!selected.length && catalog.all.length > 0 && <Typography role="status" color="text.secondary" sx={{ py: 3 }}>No benchmark graphs selected. Add benchmarks to begin.</Typography>}
        <Stack direction="row" sx={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mt: 1.5 }}>
          <Typography variant="caption" color="text.secondary">Each chart is one workload. Missing configurations remain gaps, not zero results.</Typography>
          {selectedRunIds.length > 0 && <Button size="small" color="inherit" onClick={onClearSelectedRuns}>Clear selected runs ({selectedRunIds.length})</Button>}
        </Stack>
      </SectionCard>
      <Dialog open={draft !== null} onClose={() => setDraft(null)} fullWidth maxWidth="sm" aria-labelledby="add-benchmarks-title">
        <DialogTitle id="add-benchmarks-title">Add benchmarks</DialogTitle>
        <DialogContent><Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>Choose up to eight workloads. Options follow the selected targets, suites and execution modes.</Typography><BenchmarkGridPicker allOptions={catalog.all} availableOptions={catalog.available} hiddenCount={0} selected={draft ?? []} showingAll={false} maxSelected={8} onChange={setDraft} /></DialogContent>
        <DialogActions><Button color="inherit" onClick={() => setDraft(null)}>Cancel</Button><Button variant="contained" onClick={() => { changeSelection(draft ?? []); setDraft(null); }}>Apply selection</Button></DialogActions>
      </Dialog>
      <Dialog open={Boolean(inspectedBenchmark)} disableEnforceFocus={Boolean(selectedRecord)} onClose={closeInspector} fullWidth maxWidth="sm" aria-labelledby="inspect-results-title">
        <DialogTitle id="inspect-results-title">Inspect results · {inspectedBenchmark?.name}</DialogTitle>
        <DialogContent>
          {!resultChoices.length ? <Typography color="text.secondary">No published results in this selected configuration and period.</Typography> : <ChartPointSelector label={`${inspectedBenchmark?.name} result`} actionLabel="Open result details" description="Search and select a published attempt, including failed, timed-out and unavailable results." descriptionId="inspect-result-help" options={resultChoices} selectedId={inspectedChoiceId} onSelectionChange={(id) => { setInspectedChoiceId(id); setUnavailable(null); }} hideAction onActivate={activateChoice} />}
          {unavailable && <Alert severity="info" sx={{ mt: 1 }}>Unavailable · {unavailable.target} · {unavailable.mode} · {unavailable.run.runId}. No published result; this is not a failed or zero-duration measurement.</Alert>}
        </DialogContent>
        <DialogActions sx={{ px: 3, py: 1.5, flexWrap: 'wrap', gap: 1 }}><Button color="inherit" onClick={closeInspector}>Close</Button><Button variant="contained" disabled={!inspectedChoice} onClick={() => activateChoice(inspectedChoice)}>Open result details</Button></DialogActions>
      </Dialog>
      <BenchmarkResultDialog record={selectedRecord} repository={data.repository} onClose={() => setSelectedRecord(null)} />
    </Box>
  );
}
