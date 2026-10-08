import { useMemo } from 'react';
import { Autocomplete, Box, TextField, Typography } from '@mui/material';
import { monoFont } from '../../theme/tokens';
import { filterRunOptions, runOptionLabel } from './branchPresentation';

export default function BranchRunSelector({ label, options, value, selectedId, onChange }) {
  const selected = useMemo(() => value ?? (selectedId ? { runId: selectedId, unavailable: true } : null), [value, selectedId]);
  const displayedOptions = useMemo(() => selected?.unavailable ? [selected, ...options] : options, [selected, options]);
  const optionLabel = (run) => run.unavailable ? `Unavailable attempt · ${run.runId}` : runOptionLabel(run);
  return <Box sx={{ minWidth: 0 }}>
    <Autocomplete size="small" disableClearable openOnFocus autoHighlight options={displayedOptions} value={selected}
      getOptionKey={(run) => run.runId} getOptionLabel={optionLabel}
      getOptionDisabled={(run) => Boolean(run.unavailable)}
      isOptionEqualToValue={(option, chosen) => option.runId === chosen.runId}
      filterOptions={(runs, search) => filterRunOptions(runs.filter((run) => !run.unavailable), search)}
      onChange={(_, run) => run && onChange(run.runId)}
      noOptionsText="No published attempts match this search"
      renderOption={({ key, ...props }, run) => <Box component="li" key={key} {...props} sx={{ display: 'block', py: 1, overflowWrap: 'anywhere' }}>
        <Typography variant="body2" sx={{ fontFamily: monoFont }}>{runOptionLabel(run)}</Typography>
        {run.provenance?.commitMessage && <Typography variant="caption" color="text.secondary">{run.provenance.commitMessage}</Typography>}
      </Box>}
      renderInput={(params) => <TextField {...params} label={label} placeholder={label === 'Reference' ? 'Choose reference' : 'Choose candidate'} helperText={`Search ${options.length} published ${label === 'Candidate' ? 'attempts on this branch' : 'attempts across all branches'}`} />}
      slotProps={{ listbox: { sx: { maxHeight: 360, '& li': { minHeight: 44 } } }, popper: { sx: { maxWidth: 'calc(100vw - 24px)' } }, clearIndicator: { sx: { minWidth: 44, minHeight: 44 } }, popupIndicator: { sx: { minWidth: 44, minHeight: 44 } } }}
      sx={{ '& .MuiInputBase-root': { minHeight: 44 }, minWidth: 0 }} />
    <Typography component="div" variant="caption" data-testid={`${label.toLowerCase()}-selected-identity`} sx={{ mt: 0.75, overflowWrap: 'anywhere', fontFamily: monoFont, color: 'text.secondary' }}>
      {value ? <>{value.provenance.rocjitsuCommitSha}<br />Attempt {value.runId} · {value.timestamp}</> : selectedId ? `Unavailable attempt: ${selectedId}` : 'No reference selected · Choose reference above'}
    </Typography>
  </Box>;
}
