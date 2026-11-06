import RunOptionDetails from '../shared/RunOptionDetails';
import { runSelectionLabel } from '../shared/runOptionPresentation';
import { commitTimestampFor } from '../../data/runOrdering';
import { formatDuration, formatRunTime } from '../../utils/formatters';
import { matchAnchorWidth } from '../shared/dropdownGeometry';
import {
  Autocomplete,
  Box,
  createFilterOptions,
  TextField,
  Typography,
} from '@mui/material';
import { visuallyHiddenStyles } from '../../theme/styles';

// One option per run means a grid chart already offers hundreds of choices, so the listbox renders
// a bounded window of the matches and the search narrows the full set.
const MAX_RENDERED_OPTIONS = 50;
const filterOptions = createFilterOptions({
  limit: MAX_RENDERED_OPTIONS,
  stringify: (option) => [option.label, option.run && runSelectionLabel(option.run), option.run?.branch,
    formatRunTime(commitTimestampFor(option.run)), option.run?.timestamp, commitTimestampFor(option.run),
    option.run?.provenance?.rocjitsuCommitSha].filter(Boolean).join(' '),
});

export default function ChartPointSelector({
  label,
  description,
  descriptionId,
  options,
  selectedId,
  onSelectionChange,
}) {
  if (options.length === 0) return null;

  const selected = options.find((option) => option.id === selectedId) ?? options[0];

  return (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'flex-start', gap: 1, mt: 1 }}>
      <Typography id={descriptionId} sx={visuallyHiddenStyles}>{description}</Typography>
      <Autocomplete
        size="small"
        autoHighlight
        disableClearable
        openOnFocus
        options={options}
        slotProps={{ popper: { modifiers: [matchAnchorWidth] }, listbox: { sx: { overflowWrap: 'anywhere' } } }}
        filterOptions={filterOptions}
        getOptionKey={(option) => option.id}
        getOptionLabel={(option) => option.run ? `${runSelectionLabel(option.run)} · ${option.target} · ${option.mode} · ${option.status}` : option.label}
        renderOption={({ key, ...props }, option) => <Box component="li" key={key} {...props} sx={{ py: 0.9, alignItems: 'flex-start', overflowWrap: 'anywhere' }}>
          {option.run ? <RunOptionDetails run={option.run} result={`${option.target} · ${option.mode} · ${option.status}${option.status === 'completed' ? ` · ${formatDuration(option.record?.test.durationSeconds)}` : ''}`} /> : option.label}
        </Box>}
        isOptionEqualToValue={(option, value) => option.id === value.id}
        value={selected}
        onChange={(_, option) => onSelectionChange(option?.id ?? '')}
        sx={{ minWidth: 0, width: '100%', maxWidth: 540 }}
        renderInput={(params) => (
          <TextField
            {...params}
            label={label}
            slotProps={{ ...params.slotProps, inputLabel: { ...params.slotProps.inputLabel, shrink: true } }}
            helperText={`Search all ${options.length} runs; up to ${MAX_RENDERED_OPTIONS} results shown`}
          />
        )}
      />
    </Box>
  );
}
