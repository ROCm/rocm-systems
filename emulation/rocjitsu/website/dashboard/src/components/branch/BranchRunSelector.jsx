import RunOptionDetails from '../shared/RunOptionDetails';
import { runSelectionLabel } from '../shared/runOptionPresentation';
import { useMemo } from 'react';

import { Autocomplete, Box, TextField, Typography } from '@mui/material';

import { filterRunOptions } from './branchPresentation';

export default function BranchRunSelector({ label, options, value, selectedId, onChange }) {
  const selected = useMemo(() => value ?? (selectedId ? { runId: selectedId, unavailable: true } : null), [value, selectedId]);
  const displayedOptions = useMemo(() => selected?.unavailable ? [selected, ...options] : options, [selected, options]);
  const optionLabel = (run) => run.unavailable ? `Unavailable attempt · ${run.runId}` : runSelectionLabel(run);
  return <Box sx={{ minWidth: 0 }}>
    <Autocomplete size="small" disableClearable openOnFocus autoHighlight options={displayedOptions} value={selected}
      getOptionKey={(run) => run.runId} getOptionLabel={optionLabel}
      getOptionDisabled={(run) => Boolean(run.unavailable)}
      isOptionEqualToValue={(option, chosen) => option.runId === chosen.runId}
      filterOptions={(runs, search) => filterRunOptions(runs.filter((run) => !run.unavailable), search)}
      onChange={(_, run) => run && onChange(run.runId)}
      noOptionsText="No published attempts match this search"
      renderOption={({ key, ...props }, run) => <Box component="li" key={key} {...props} sx={{ py: '7px', alignItems: 'flex-start', overflowWrap: 'anywhere' }}>
        <RunOptionDetails run={run} />
      </Box>}
      renderInput={(params) => <TextField {...params} label={label} slotProps={{ ...params.slotProps, inputLabel: { ...params.slotProps.inputLabel, shrink: true } }} placeholder={label === 'Reference' ? 'Choose reference' : 'Choose candidate'} />}
      slotProps={{ listbox: { sx: { maxHeight: 360, '& li': { minHeight: 44 } } }, popper: { sx: { maxWidth: 'calc(100vw - 24px)' } }, clearIndicator: { sx: { minWidth: 44, minHeight: 44 } }, popupIndicator: { sx: { minWidth: 44, minHeight: 44 } } }}
      sx={{ '& .MuiInputBase-root': { minHeight: 44 }, ...(selected?.unavailable ? { '& .MuiOutlinedInput-notchedOutline': { borderColor: 'warning.main' } } : {}), minWidth: 0 }} />
    {!value && <Typography component="div" variant="caption" sx={{ mt: '6px', overflowWrap: 'anywhere', color: 'text.secondary' }}>
      {selectedId ? `Unavailable attempt: ${selectedId}` : 'No reference selected · Choose reference above'}
    </Typography>}
  </Box>;
}
