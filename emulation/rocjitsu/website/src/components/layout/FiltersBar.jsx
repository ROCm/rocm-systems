import { useLayoutEffect, useRef, useState } from 'react';
import {
  Autocomplete,
  Box,
  Checkbox,
  Chip,
  ListItemText,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import TuneRoundedIcon from '@mui/icons-material/TuneRounded';

const CHECK_ALL = '__check_all__';
const TAG_EDGE_GAP = 12;

function ResponsiveTags({ values, getItemProps }) {
  const containerRef = useRef(null);
  const [visibleCount, setVisibleCount] = useState(values.length);

  useLayoutEffect(() => {
    const root = containerRef.current?.closest('.MuiAutocomplete-root');
    if (!root || typeof ResizeObserver === 'undefined') return undefined;

    const calculateVisibleCount = () => {
      const inputRoot = root.querySelector('.MuiAutocomplete-inputRoot');
      const endAdornment = root.querySelector('.MuiAutocomplete-endAdornment');
      const sampleChip = root.querySelector('[data-tag-measurement]');
      if (!inputRoot || !endAdornment || !sampleChip) return;

      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      const sampleLabel = sampleChip.querySelector('.MuiChip-label');
      const labelStyle = window.getComputedStyle(sampleLabel);
      context.font = `${labelStyle.fontWeight} ${labelStyle.fontSize} ${labelStyle.fontFamily}`;
      const letterSpacing = Number.parseFloat(labelStyle.letterSpacing) || 0;
      const measureText = (text) => context.measureText(text).width + Math.max(0, text.length - 1) * letterSpacing;
      const chipStyle = window.getComputedStyle(sampleChip);
      const chipMargins = (Number.parseFloat(chipStyle.marginLeft) || 0) + (Number.parseFloat(chipStyle.marginRight) || 0);
      const chipExtra = sampleChip.getBoundingClientRect().width + chipMargins - measureText(sampleLabel.textContent ?? '');
      const overflowTag = root.querySelector('[data-overflow-tag]');
      const overflowExtra = overflowTag
        ? overflowTag.getBoundingClientRect().width - measureText(overflowTag.textContent ?? '')
        : 8;
      const inputBounds = inputRoot.getBoundingClientRect();
      const endBounds = endAdornment.getBoundingClientRect();
      const availableWidth = Math.max(0, endBounds.left - inputBounds.left - TAG_EDGE_GAP);
      const tagWidths = values.map((value) => measureText(value) + chipExtra);
      const allWidth = tagWidths.reduce((total, width) => total + width, 0);

      let nextCount = values.length;
      if (allWidth > availableWidth) {
        nextCount = 0;
        let visibleWidth = 0;
        for (let count = 1; count < values.length; count += 1) {
          visibleWidth += tagWidths[count - 1];
          const hiddenCount = values.length - count;
          const totalWidth = visibleWidth + measureText(`+${hiddenCount}`) + overflowExtra;
          if (totalWidth <= availableWidth) nextCount = count;
        }
      }
      setVisibleCount((current) => current === nextCount ? current : nextCount);
    };

    calculateVisibleCount();
    const observer = new ResizeObserver(calculateVisibleCount);
    observer.observe(root);
    return () => observer.disconnect();
  }, [values]);

  const hiddenCount = values.length - visibleCount;
  return (
    <Box component="span" ref={containerRef} sx={{ display: 'contents' }}>
      {values.length > 0 && (
        <Chip
          label={values[0]}
          size="small"
          className="MuiAutocomplete-tag"
          onDelete={() => {}}
          data-tag-measurement="true"
          aria-hidden="true"
          sx={{ position: 'absolute', visibility: 'hidden', pointerEvents: 'none', height: 23 }}
        />
      )}
      {values.slice(0, visibleCount).map((option, index) => (
        <Chip
          label={option}
          size="small"
          {...getItemProps({ index })}
          data-responsive-tag="true"
          key={option}
          sx={{ height: 23, flexShrink: 0 }}
        />
      ))}
      {hiddenCount > 0 && (
        <Box
          component="span"
          data-overflow-tag="true"
          title={values.slice(visibleCount).join(', ')}
          aria-label={`${hiddenCount} more selected`}
          sx={{ px: 0.5, flexShrink: 0, color: 'text.secondary', fontSize: 12, fontWeight: 750 }}
        >
          +{hiddenCount}
        </Box>
      )}
    </Box>
  );
}

function MultiSelect({ label, options, value, onChange }) {
  const allSelected = options.length > 0 && value.length === options.length;
  const someSelected = value.length > 0 && !allSelected;
  const groupName = label.toLowerCase();

  return (
    <Box data-testid={`${groupName}-filter`} sx={{ minWidth: 0 }}>
      <Autocomplete
        multiple
        disableCloseOnSelect
        options={[CHECK_ALL, ...options]}
        value={value}
        getOptionLabel={(option) => option === CHECK_ALL ? `Check all ${groupName}` : option}
        filterOptions={(availableOptions, { inputValue }) => availableOptions.filter((option) => (
          option === CHECK_ALL || option.toLowerCase().includes(inputValue.trim().toLowerCase())
        ))}
        onChange={(_, nextValue, reason, details) => {
          if (details?.option === CHECK_ALL) {
            onChange(allSelected ? [] : options);
            return;
          }
          onChange(reason === 'clear' ? [] : nextValue.filter((option) => option !== CHECK_ALL));
        }}
        renderOption={(props, option, { selected }) => {
          const { key, ...optionProps } = props;
          const isCheckAll = option === CHECK_ALL;
          return (
            <Box component="li" key={key} {...optionProps} sx={{ py: 0.65, borderBottom: isCheckAll ? 1 : 0, borderColor: 'divider' }}>
              <Checkbox
                size="small"
                checked={isCheckAll ? allSelected : selected}
                indeterminate={isCheckAll && someSelected}
                tabIndex={-1}
                disableRipple
                sx={{ mr: 1, p: 0.35 }}
              />
              <ListItemText
                primary={isCheckAll ? `Check all ${groupName}` : option}
                secondary={isCheckAll ? `${value.length} of ${options.length} selected` : null}
                slotProps={{ primary: { variant: 'body2', fontWeight: isCheckAll ? 700 : 500 }, secondary: { variant: 'caption' } }}
              />
            </Box>
          );
        }}
        renderValue={(tagValue, getItemProps) => (
          <ResponsiveTags values={tagValue} getItemProps={getItemProps} />
        )}
        renderInput={(params) => <TextField {...params} label={label} size="small" />}
        sx={{
          minWidth: 0,
          '& .MuiAutocomplete-inputRoot': { flexWrap: 'nowrap', overflow: 'hidden' },
          '& .MuiAutocomplete-input': { minWidth: '4px !important' },
        }}
      />
    </Box>
  );
}

export default function FiltersBar({ data, state }) {
  return (
    <Paper variant="outlined" sx={{ p: { xs: 1.5, md: 2 }, borderRadius: 3, boxShadow: 1 }}>
      <Stack direction="row" sx={{ alignItems: 'center', gap: 1, mb: 1.5 }}>
        <TuneRoundedIcon color="primary" sx={{ fontSize: 18 }} />
        <Typography variant="overline" color="text.secondary">Global filters</Typography>
        <Typography variant="caption" color="text.disabled" sx={{ ml: 'auto', display: { xs: 'none', md: 'block' } }}>
          Applies to every dashboard view
        </Typography>
      </Stack>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))', lg: 'minmax(280px, 360px) minmax(360px, 460px)' }, gap: 1.25 }}>
        <MultiSelect label="Targets" options={data.targets} value={state.targets} onChange={state.setTargets} />
        <MultiSelect label="Suites" options={data.suites} value={state.suites} onChange={state.setSuites} />
      </Box>
    </Paper>
  );
}
