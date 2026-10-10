import {
  Autocomplete,
  Box,
  Button,
  Chip,
  ListItemText,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import CheckBoxIcon from '@mui/icons-material/CheckBox';
import CheckBoxOutlineBlankIcon from '@mui/icons-material/CheckBoxOutlineBlank';

import { boundedGridSelection, filterBenchmarkOptions } from './benchmarkExplorer';

// Size before Popper's flip/overflow calculations, using the larger side of the
// anchor. A fixed 340px list can extend below a short dialog/phone viewport.
const benchmarkViewport = {
  name: 'benchmarkViewport', enabled: true, phase: 'beforeRead',
  fn: ({ state }) => {
    const anchor = state.elements.reference;
    const window = anchor.ownerDocument.defaultView;
    const viewport = window.visualViewport;
    const top = viewport?.offsetTop ?? 0;
    const bottom = top + (viewport?.height ?? window.innerHeight);
    const rect = anchor.getBoundingClientRect();
    const height = Math.max(0, Math.min(340, Math.max(rect.top - top, bottom - rect.bottom) - 8));
    state.elements.popper.style.setProperty('--benchmark-listbox-height', `${height}px`);
    state.rects.popper.height = state.elements.popper.offsetHeight;
  },
};

function formatDataType(value) {
  if (value == null) return null;
  return typeof value === 'string' ? value.toUpperCase() : String(value);
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
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {showingAll
              ? `Showing all ${totalCount} benchmarks`
              : `${hiddenCount} benchmark${hiddenCount === 1 ? '' : 's'} hidden by global target / suite filters`}
          </Typography>
          <Button size="small" onClick={onToggleScope} sx={{ flexShrink: 0 }}>
            {showingAll ? 'Use global target / suite filters' : 'Show all benchmarks'}
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
  // MUI groupBy requires adjacent suite members; publication order can interleave.
  const options = [...(showingAll ? allOptions : availableOptions)].sort((a, b) => a.suite.localeCompare(b.suite));

  return (
    <Autocomplete
      multiple
      disableCloseOnSelect
      openOnFocus
      size="small"
      // Keep chip geometry stable on blur so a pointer click below the picker
      // cannot miss its target when the input loses focus.
      limitTags={-1}
      options={options}
      value={selected}
      getOptionKey={(option) => option.id}
      getOptionLabel={(option) => option.name}
      isOptionEqualToValue={(option, value) => option.id === value.id}
      groupBy={(option) => option.suite}
      filterOptions={(candidateOptions, state) => filterBenchmarkOptions(candidateOptions, state.inputValue)}
      getOptionDisabled={(option) => selected.length >= maxSelected && !selectedIds.has(option.id)}
      onChange={(_, nextOptions) => {
        onChange(boundedGridSelection(nextOptions, maxSelected));
      }}
      noOptionsText={showingAll ? 'No benchmarks match your search' : 'No benchmarks match within the selected targets / suites'}
      renderOption={(props, option, state) => {
        const { key, ...optionProps } = props;
        const outsideFilter = !availableIds.has(option.id);
        return (
          <Box component="li" key={key} {...optionProps} sx={{ gap: 1, py: 0.75 }}>
            <Box component="span" aria-hidden="true" sx={{ display: 'inline-flex', p: 0.25, pointerEvents: 'none', color: state.selected ? 'primary.main' : 'text.secondary' }}>
              {state.selected ? <CheckBoxIcon fontSize="small" /> : <CheckBoxOutlineBlankIcon fontSize="small" />}
            </Box>
            <ListItemText
              primary={option.name}
              secondary={[option.problem?.operation, formatDataType(option.problem?.dataType)].filter(Boolean).join(' · ')}
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
          helperText={`${selected.length} of ${maxSelected} benchmarks selected · ${availableOptions.length} available in the selected targets / suites · Search all; up to 50 matches shown`}
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
        popper: { modifiers: [benchmarkViewport, { name: 'flip', options: { padding: 8 } }, { name: 'preventOverflow', options: { padding: 8 } }] },
        listbox: { sx: { maxHeight: 'min(340px, var(--benchmark-listbox-height, calc(100dvh - 16px)))', overflowY: 'auto', overscrollBehavior: 'contain', '& .MuiAutocomplete-option': { overflowWrap: 'anywhere' } } },
      }}
      sx={{ width: '100%', maxWidth: 720, '& .MuiAutocomplete-inputRoot': { maxHeight: 'min(240px, 30dvh)', overflowY: 'auto', overscrollBehavior: 'contain' }, '& .MuiAutocomplete-tag': { maxWidth: 'calc(100% - 6px)' } }}
    />
  );
}
