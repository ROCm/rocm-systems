import { useEffect, useRef, useState } from 'react';
import { Box, Button, Paper, TextField, Typography } from '@mui/material';
import { compareRunExecution } from '../../data/runOrdering';
import { selectAutomaticReference, selectPublishedBranches } from '../../data/branchSelectors';
import BranchList from '../branch/BranchList';
import BranchPicker from '../branch/BranchPicker';
import BranchDetail from '../branch/BranchDetail';
import { resolveBranchSelection, selectionForAction, selectionForBranch, selectionForCandidate } from '../branch/branchSelection';

export default function BranchRunsView({ data, state, onOpenComparison }) {
  const [query, setQuery] = useState('');
  const [pickerCollapsed, setPickerCollapsed] = useState(false);
  const [pr, setPr] = useState('all');
  const scrollRef = useRef(null);
  const headingRef = useRef(null);
  const listPosition = useRef(0);
  const detail = Boolean(state?.branchSelection?.detail);
  const previousDetail = useRef(detail);
  // Restore UI scroll/focus only; defaults never write route history.
  useEffect(() => {
    if (!detail && scrollRef.current) {
      scrollRef.current.scrollTop = listPosition.current;
      if (previousDetail.current) (scrollRef.current.querySelector('[aria-pressed="true"]') ?? scrollRef.current).focus({ preventScroll: true });
    }
    if (detail && !previousDetail.current) headingRef.current?.focus({ preventScroll: true });
    previousDetail.current = detail;
  }, [detail]);
  if (!data) return <Paper variant="outlined" sx={{ p: 3 }}><Typography role="status" variant="body2">Loading published branch runs…</Typography></Paper>;
  const branches = selectPublishedBranches(data);
  const visible = selectPublishedBranches(data, { query, pr });
  const runs = [...(data.allRuns ?? [])].sort(compareRunExecution).reverse();
  const snapshot = state?.branchSelection ?? {};
  const preliminary = resolveBranchSelection(snapshot, { branches, runs });
  const candidate = runs.find((run) => run.runId === preliminary.candidateId && run.branch === preliminary.branch) ?? null;
  const automatic = selectAutomaticReference(data, candidate);
  const selection = resolveBranchSelection(snapshot, { branches, runs, automaticReferenceId: automatic.run?.runId });
  const reference = runs.find((run) => run.runId === selection.referenceId) ?? null;
  const candidateOptions = runs.filter((run) => run.branch === selection.branch);
  const clear = () => { setQuery(''); setPr('all'); };
  const write = (next) => state.setBranchSelection(next);
  return <Box data-testid="branch-runs" data-screen={selection.detail ? 'detail' : 'list'} sx={{ minWidth: 0, containerType: 'inline-size', containerName: 'branch-runs',
    '@media (max-width:767px)': {
      '&[data-screen="detail"] .branch-toolbar, &[data-screen="detail"] .branch-picker': { display: 'none' },
      '&[data-screen="list"] .branch-detail': { display: 'none' },
    },
  }}>
    <Typography variant="body2" color="text.secondary">Published branch benchmarks · all branches are public</Typography>
    {branches.length || selection.branch ? <>
      <Box className="branch-toolbar" sx={{ display: 'flex', gap: 1.5, my: 2, flexWrap: 'wrap', alignItems: 'center' }}>
        <TextField type="search" label="Search branch, PR or SHA" placeholder="Search branch, PR or SHA" size="small" value={query} onChange={(event) => setQuery(event.target.value)} sx={{ flex: '1 1 240px', '& .MuiInputBase-root': { minHeight: 44 }, '& .MuiInputBase-input': { py: '10px' }, '& .MuiInputLabel-root:not(.MuiInputLabel-shrink)': { transform: 'translate(14px, 12px) scale(1)' } }} />
        <TextField select label="Pull request" size="small" value={pr} onChange={(event) => setPr(event.target.value)} slotProps={{ select: { native: true }, inputLabel: { shrink: true } }} sx={{ width: 220, flexGrow: { xs: 1, sm: 0 }, '& .MuiInputBase-root': { minHeight: 44 } }}>
          <option value="all">All branches</option><option value="with-pr">With PR</option><option value="no-pr">No PR</option>
        </TextField>
        {(query || pr !== 'all') && <Button onClick={clear} sx={{ minHeight: 44 }}>Clear filters</Button>}
      </Box>
      <Box sx={{ display: 'grid', gap: 2, minWidth: 0, gridTemplateColumns: 'minmax(0,1fr)', '@container branch-runs (min-width:1000px)': { gridTemplateColumns: pickerCollapsed ? '44px minmax(0,1fr)' : '304px minmax(0,1fr)', alignItems: 'stretch' } }}>
        <BranchPicker collapsed={pickerCollapsed} onToggle={() => setPickerCollapsed((collapsed) => !collapsed)}>
        <BranchList branches={visible} selectedBranch={selection.branch} scrollRef={scrollRef} onClear={clear}
          onScroll={(event) => { if (!selection.detail || event.currentTarget.clientHeight > 0) listPosition.current = event.currentTarget.scrollTop; }}
          onSelect={(entry) => {
            listPosition.current = scrollRef.current?.scrollTop ?? listPosition.current;
            write(selectionForBranch(selection, entry, selectAutomaticReference(data, entry.latestRun).run?.runId));
          }} />
        </BranchPicker>
        <BranchDetail selection={selection} candidate={candidate} reference={reference} automatic={automatic} candidateOptions={candidateOptions} runs={runs} onOpenComparison={onOpenComparison} repository={data.repository} headingRef={headingRef}
          onCandidate={(runId) => write(selectionForCandidate(selection, runId, selectAutomaticReference(data, runs.find((run) => run.runId === runId)).run?.runId))}
          onReference={(runId) => write(selectionForAction(selection, { type: 'reference', runId }))}
          onAutomatic={() => write(selectionForAction(selection, { type: 'automatic', runId: automatic.run?.runId }))}
          onConfiguration={(target, mode) => write(selectionForAction(selection, { type: 'configuration', target, mode }))}
          onBack={() => write(selectionForAction(selection, { type: 'back' }))} />
      </Box>
    </> : <Paper variant="outlined" sx={{ mt: 2, p: 3 }}>
      <Typography variant="h6" component="h2">No published branch runs</Typography>
      <Typography variant="body2" color="text.secondary">Only branches active within 30 days of the latest published data are shown.</Typography>
    </Paper>}
  </Box>;
}
