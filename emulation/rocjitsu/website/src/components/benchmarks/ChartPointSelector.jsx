import { useState } from 'react';
import {
  Autocomplete,
  Box,
  Button,
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
  actionLabel,
  description,
  descriptionId,
  options,
  onActivate,
}) {
  const [requestedId, setRequestedId] = useState('');

  if (options.length === 0) return null;

  const selected = options.find((option) => option.id === requestedId) ?? options[0];

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
        onChange={(_, option) => setRequestedId(option?.id ?? '')}
        sx={{ minWidth: 260, maxWidth: '100%' }}
        renderInput={(params) => (
          <TextField
            {...params}
            label={label}
            helperText={`Search all ${options.length} runs; up to ${MAX_RENDERED_OPTIONS} results shown`}
          />
        )}
      />
      <Button size="small" variant="outlined" sx={{ mt: 0.3 }} onClick={() => onActivate(selected)}>{actionLabel}</Button>
    </Box>
  );
}
