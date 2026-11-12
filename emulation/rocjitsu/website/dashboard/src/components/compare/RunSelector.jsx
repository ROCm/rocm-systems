import RunOptionDetails from '../shared/RunOptionDetails';
import { runSearchFields, runSelectionLabelWithBranch as optionLabel } from '../shared/runOptionPresentation';
import {
  Autocomplete,
  Box,
  TextField,
} from '@mui/material';
const MAX_VISIBLE_OPTIONS = 50;

function filterRunOptions(options, { inputValue }) {
  const query = inputValue.trim().toLowerCase();
  const matches = query
    ? options.filter((run) => runSearchFields(run).filter(Boolean).join(' ').toLowerCase().includes(query))
    : options;
  return matches.slice(0, MAX_VISIBLE_OPTIONS);
}

export default function RunSelector({ label, options, value, onChange }) {
  return (
    <Box sx={{ minWidth: 0 }}>
    <Autocomplete
      sx={{ minWidth: 0 }}
      disabled={options.length === 0}
      disableClearable
      openOnFocus
      autoHighlight
      size="small"
      options={options}
      value={value}
      getOptionKey={(run) => run.runId}
      getOptionLabel={optionLabel}
      isOptionEqualToValue={(option, selected) => option.runId === selected.runId}
      filterOptions={filterRunOptions}
      onChange={(_, run) => run && onChange(run.runId)}
      noOptionsText="No runs match this search"
      renderOption={(props, run) => {
        const { key, ...optionProps } = props;
        return (
          <Box component="li" key={key} {...optionProps} sx={{ py: '7px', alignItems: 'flex-start' }}>
            <RunOptionDetails run={run} />
          </Box>
        );
      }}
      renderInput={(params) => (
        <TextField
          {...params}
          label={label}
          // Keep empty controls in the same label/notch state as selected ones.
          // Preserve Autocomplete's anchor, input handlers and adornment slots.
          slotProps={{
            ...params.slotProps,
            inputLabel: { ...params.slotProps.inputLabel, shrink: true },
          }}
        />
      )}
      slotProps={{
        listbox: { sx: { maxHeight: 360, '& li': { overflowWrap: 'anywhere' } } },
        popper: { sx: { maxWidth: 'calc(100vw - 24px)' } },
      }}
    />
    </Box>
  );
}
