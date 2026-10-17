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
const filterOptions = createFilterOptions({ limit: MAX_RENDERED_OPTIONS });

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
        filterOptions={filterOptions}
        getOptionLabel={(option) => option.label}
        isOptionEqualToValue={(option, value) => option.id === value.id}
        value={selected}
        onChange={(_, option) => onSelectionChange(option?.id ?? '')}
        sx={{ minWidth: 0, width: '100%', maxWidth: 540 }}
        renderInput={(params) => (
          <TextField
            {...params}
            label={label}
            helperText={`Search all ${options.length} runs; up to ${MAX_RENDERED_OPTIONS} results shown`}
          />
        )}
      />
    </Box>
  );
}
