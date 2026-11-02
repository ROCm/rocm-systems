import { matchAnchorWidth } from '../shared/dropdownGeometry';
import {
  Autocomplete,
  Box,
  ListItemText,
  Paper,
  TextField,
} from '@mui/material';
import CheckBoxIcon from '@mui/icons-material/CheckBox';
import CheckBoxOutlineBlankIcon from '@mui/icons-material/CheckBoxOutlineBlank';

import { boundedGridSelection, filterBenchmarkOptions } from './benchmarkExplorer';
import { MAX_BENCHMARK_CHARTS } from '../../config/benchmarkGrid';

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

export function BenchmarkGridPicker({
  availableOptions,
  selected,
  maxSelected = MAX_BENCHMARK_CHARTS,
  onChange,
}) {
  const maximum = Math.min(maxSelected, MAX_BENCHMARK_CHARTS);
  const selectedIds = new Set(selected.map((option) => option.id));
  // MUI groupBy requires adjacent suite members; publication order can interleave.
  const options = [...availableOptions].sort((a, b) => a.suite.localeCompare(b.suite));

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
      getOptionDisabled={(option) => selected.length >= maximum && !selectedIds.has(option.id)}
      onChange={(_, nextOptions) => {
        onChange(boundedGridSelection(nextOptions, maximum));
      }}
      noOptionsText="No benchmarks match within the selected targets / suites"
      renderOption={(props, option, state) => {
        const { key, ...optionProps } = props;
        return (
          <Box component="li" key={key} {...optionProps} sx={{ gap: 1, py: 0.75 }}>
            <Box component="span" aria-hidden="true" sx={{ display: 'inline-flex', p: 0.25, pointerEvents: 'none', color: state.selected ? 'primary.main' : 'text.secondary' }}>
              {state.selected ? <CheckBoxIcon fontSize="small" /> : <CheckBoxOutlineBlankIcon fontSize="small" />}
            </Box>
            <ListItemText
              primary={option.name}
              secondary={<><Box component="span" sx={{ display: 'block', overflowWrap: 'anywhere' }}>{option.id}</Box>{[option.operation, formatDataType(option.problem?.dtype)].filter(Boolean).join(' · ')}</>}
              slotProps={{ primary: { variant: 'body2', fontWeight: 650 }, secondary: { variant: 'caption' } }}
            />
          </Box>
        );
      }}
      renderInput={(params) => (
        <TextField
          {...params}
          label="Benchmarks to graph"
          placeholder={selected.length === 0 ? 'Search and select benchmarks…' : ''}
          helperText={`${selected.length} of ${maximum} benchmarks selected · ${availableOptions.length} available in the selected targets / suites · Search all; up to 50 matches shown`}
        />
      )}
      slots={{ paper: Paper }}
      slotProps={{
        popper: { modifiers: [matchAnchorWidth, benchmarkViewport, { name: 'flip', options: { padding: 8 } }, { name: 'preventOverflow', options: { padding: 8 } }] },
        listbox: { sx: { maxHeight: 'min(340px, var(--benchmark-listbox-height, calc(100dvh - 16px)))', overflowY: 'auto', overscrollBehavior: 'contain', '& .MuiAutocomplete-option': { overflowWrap: 'anywhere' } } },
      }}
      sx={{ width: '100%', maxWidth: 720, '& .MuiAutocomplete-inputRoot': { maxHeight: 'min(240px, 30dvh)', overflowY: 'auto', overscrollBehavior: 'contain' }, '& .MuiAutocomplete-tag': { maxWidth: 'calc(100% - 6px)' } }}
    />
  );
}
