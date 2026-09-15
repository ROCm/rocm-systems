import {
  Autocomplete,
  Box,
  Button,
  Checkbox,
  Chip,
  ListItemText,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material';

function matchesSearch(option, query) {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) return true;
  return [option.name, option.suite, option.problem?.operation, option.problem?.dataType]
    .filter(Boolean)
    .join(' ')
    .toLowerCase()
    .includes(normalizedQuery);
}

function BenchmarkPickerPaper({
  children,
  hiddenCount,
  showingAll,
  totalCount,
  onToggleScope,
  ...paperProps
}) {
  return (
    <Paper {...paperProps}>
      {children}
      {hiddenCount > 0 && (
        <Stack
          direction="row"
          onMouseDown={(event) => event.preventDefault()}
          sx={{ alignItems: 'center', justifyContent: 'space-between', gap: 1, px: 1.5, py: 1, borderTop: 1, borderColor: 'divider', bgcolor: 'action.hover' }}
        >
          <Typography variant="caption" color="text.secondary">
            {showingAll
              ? `Showing all ${totalCount} benchmarks`
              : `${hiddenCount} benchmark${hiddenCount === 1 ? '' : 's'} hidden by global Suite filters`}
          </Typography>
          <Button size="small" onClick={onToggleScope} sx={{ flexShrink: 0 }}>
            {showingAll ? 'Use global Suite filters' : 'Show all benchmarks'}
          </Button>
        </Stack>
      )}
    </Paper>
  );
}

export function BenchmarkGridPicker({
  allOptions,
  availableOptions,
  hiddenCount,
  selected,
  showingAll,
  maxSelected = 8,
  onChange,
  onToggleScope,
}) {
  const availableIds = new Set(availableOptions.map((option) => option.id));
  const selectedIds = new Set(selected.map((option) => option.id));
  const options = showingAll ? allOptions : availableOptions;

  return (
    <Autocomplete
      multiple
      disableCloseOnSelect
      openOnFocus
      size="small"
      limitTags={2}
      options={options}
      value={selected}
      getOptionKey={(option) => option.id}
      getOptionLabel={(option) => option.name}
      isOptionEqualToValue={(option, value) => option.id === value.id}
      groupBy={(option) => option.suite}
      filterOptions={(candidateOptions, state) => candidateOptions.filter((option) => matchesSearch(option, state.inputValue))}
      getOptionDisabled={(option) => selected.length >= maxSelected && !selectedIds.has(option.id)}
      onChange={(_, nextOptions) => {
        if (nextOptions.length <= maxSelected) onChange(nextOptions);
      }}
      noOptionsText={showingAll ? 'No benchmarks match your search' : 'No benchmarks match within the selected suites'}
      renderOption={(props, option, state) => {
        const { key, ...optionProps } = props;
        const outsideFilter = !availableIds.has(option.id);
        return (
          <Box component="li" key={key} {...optionProps} sx={{ gap: 1, py: 0.75 }}>
            <Checkbox checked={state.selected} size="small" sx={{ p: 0.25 }} />
            <ListItemText
              primary={option.name}
              secondary={[option.problem?.operation, option.problem?.dataType?.toUpperCase()].filter(Boolean).join(' · ')}
              slotProps={{ primary: { variant: 'body2', fontWeight: 650 }, secondary: { variant: 'caption' } }}
            />
            {outsideFilter && <Chip size="small" label="Outside filter" color="warning" variant="outlined" />}
          </Box>
        );
      }}
      renderInput={(params) => (
        <TextField
          {...params}
          label="Benchmarks to graph"
          placeholder={selected.length === 0 ? 'Search and select benchmarks…' : ''}
          helperText={`${selected.length} of ${maxSelected} benchmarks selected · ${availableOptions.length} available in the selected suites`}
        />
      )}
      slots={{ paper: BenchmarkPickerPaper }}
      slotProps={{
        paper: {
          hiddenCount,
          showingAll,
          totalCount: allOptions.length,
          onToggleScope,
        },
        listbox: { sx: { maxHeight: 340 } },
      }}
      sx={{ width: '100%', maxWidth: 720 }}
    />
  );
}

export default function BenchmarkPicker({
  allOptions,
  availableOptions,
  hiddenCount,
  selected,
  showingAll,
  onChange,
  onToggleScope,
}) {
  const availableIds = new Set(availableOptions.map((option) => option.id));
  const options = showingAll ? allOptions : availableOptions;

  return (
    <Autocomplete
      disableClearable
      openOnFocus
      size="small"
      options={options}
      value={selected}
      getOptionKey={(option) => option.id}
      getOptionLabel={(option) => option.name}
      isOptionEqualToValue={(option, value) => option.id === value.id}
      groupBy={(option) => option.suite}
      filterOptions={(candidateOptions, state) => candidateOptions.filter((option) => matchesSearch(option, state.inputValue))}
      onChange={(_, nextOption) => onChange(nextOption)}
      noOptionsText={showingAll ? 'No benchmarks match your search' : 'No benchmarks match within the selected suites'}
      renderOption={(props, option) => {
        const { key, ...optionProps } = props;
        const outsideFilter = !availableIds.has(option.id);
        return (
          <Box component="li" key={key} {...optionProps} sx={{ gap: 1, py: 0.75 }}>
            <ListItemText
              primary={option.name}
              secondary={[option.problem?.operation, option.problem?.dataType?.toUpperCase()].filter(Boolean).join(' · ')}
              slotProps={{ primary: { variant: 'body2', fontWeight: 650 }, secondary: { variant: 'caption' } }}
            />
            {outsideFilter && <Chip size="small" label="Outside filter" color="warning" variant="outlined" />}
          </Box>
        );
      }}
      renderInput={(params) => (
        <TextField
          {...params}
          label="Benchmark"
          placeholder="Search benchmarks…"
          helperText={`${availableOptions.length} benchmarks available in the selected suites`}
        />
      )}
      slots={{ paper: BenchmarkPickerPaper }}
      slotProps={{
        paper: {
          hiddenCount,
          showingAll,
          totalCount: allOptions.length,
          onToggleScope,
        },
        listbox: { sx: { maxHeight: 340 } },
      }}
      sx={{ width: { xs: '100%', sm: 420 }, maxWidth: '100%' }}
    />
  );
}
