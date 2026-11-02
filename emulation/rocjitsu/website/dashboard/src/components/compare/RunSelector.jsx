import { monoFont } from '../../theme/tokens';
import RunOptionDetails from '../shared/RunOptionDetails';
import { runSearchFields, runSelectionLabel as optionLabel } from '../shared/runOptionPresentation';
import { matchAnchorWidth } from '../shared/dropdownGeometry';
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
          <Box component="li" key={key} {...optionProps} sx={{ py: 0.9, alignItems: 'flex-start' }}>
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
          helperText={`Search ${options.length} runs · Showing up to ${MAX_VISIBLE_OPTIONS} matches`}
        />
      )}
      slotProps={{
        listbox: { sx: { maxHeight: 360, '& li': { overflowWrap: 'anywhere' } } },
        popper: { modifiers: [matchAnchorWidth], sx: { maxWidth: 'calc(100vw - 24px)' } },
      }}
    />
    {/* Reserve both identity rows before selection so stacked controls do not move.
        Keep full text in the DOM and titles when a long identity is ellipsized. */}
    <Box
      data-testid={value ? `${label.toLowerCase().replaceAll(' ', '-')}-selected-identity` : undefined}
      aria-hidden={!value || undefined}
      sx={{ mt: 0.75, px: 0.5, fontFamily: monoFont, fontSize: 11, lineHeight: '16px', color: 'text.secondary' }}
    >
      <Box title={value?.runId} sx={{ height: 16, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{value?.runId}</Box>
      <Box title={value ? `${value.branch} · ${(value.modes ?? []).join(' / ')}` : undefined}
        sx={{ mt: 0.25, height: 16, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {value && `${value.branch} · ${(value.modes ?? []).join(' / ')}`}
      </Box>
    </Box>
    </Box>
  );
}
