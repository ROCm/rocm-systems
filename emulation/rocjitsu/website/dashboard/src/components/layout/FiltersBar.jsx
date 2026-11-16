import { fontSizes, fontWeights, textStyles } from '../../theme/typography';
import { Box, ButtonBase, Checkbox, FormControlLabel, Typography } from '@mui/material';
import { CategoryMarker } from '../shared/CategoryTag';

function FilterGroup({ label, options, value, onChange, disabled, kind }) {
  return (
    <Box role="group" aria-label={label} data-testid={`${label.toLowerCase().replaceAll(' ', '-')}-filter`} sx={{ borderTop: 1, borderColor: 'dashboard.border', m: 0, pt: 'var(--sidebar-group-top, 12px)', pb: 'var(--sidebar-group-bottom, 8px)', minWidth: 0 }}>
      <Typography variant="overline" sx={{ display: 'block', mb: '6px', height: '16px', ...textStyles.captionCompactEmphasized, color: 'text.secondary' }}>{label}</Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: '7px', rowGap: 'var(--sidebar-chip-row-gap, 7px)', pt: '2px' }}>
        {options.map((option) => (
          <FormControlLabel key={option} disabled={disabled} sx={{
            m: 0, position: 'relative', display: 'inline-flex', boxSizing: 'border-box', height: '30px', padding: '5px 12px 5px 10px', borderRadius: '999px', border: '1px solid',
            borderColor: value.includes(option) ? 'dashboard.selectionBorder' : 'dashboard.controlBorder',
            borderStyle: value.includes(option) ? 'solid' : 'dashed',
            bgcolor: value.includes(option) ? 'action.selected' : 'transparent', color: disabled ? 'text.disabled' : value.includes(option) ? 'dashboard.sidebarAccentText' : 'text.muted',
            boxShadow: value.includes(option) ? (theme) => `inset 0 0 0 1px ${theme.palette.dashboard.selectionBorder}` : 'none',
            '&:has(input:focus-visible)': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: 2 },
            '& .MuiFormControlLabel-label.Mui-disabled': { color: 'text.disabled' },
          }}
            control={<Checkbox size="small" disabled={disabled} checked={value.includes(option)}
              onChange={() => onChange(value.includes(option) ? value.filter((item) => item !== option) : [...value, option])}
              sx={{ position: 'absolute', inset: 0, width: '100%', height: '100%', p: 0, opacity: 0 }} />}
            label={<Box component="span" sx={{ display: 'flex', alignItems: 'center', gap: '7px', fontSize: fontSizes.body, lineHeight: '18px', fontWeight: fontWeights.emphasized, whiteSpace: 'nowrap' }}><CategoryMarker label={option} kind={kind} size={9} muted={!value.includes(option)} />{option}</Box>} />
        ))}
      </Box>
      {!options.length && <Typography variant="caption" color="text.secondary">No available {label.toLowerCase()}</Typography>}
      {kind === 'mode' && options.length > 0 && <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: '4px', pt: '8px', px: '2px', ...textStyles.caption }}>single · single-threaded<br />default · automatic policy</Typography>}
    </Box>
  );
}

export default function FiltersBar({ data, state, disabled = false }) {
  const groups = [
    { options: data.targets ?? [], value: state.targets ?? [], onChange: state.setTargets },
    { options: data.suites ?? [], value: state.suites ?? [], onChange: state.setSuites },
    { options: data.modes ?? [], value: state.modes ?? [], onChange: state.setModes },
  ];
  const totalCount = groups.reduce((total, { options }) => total + options.length, 0);
  const selectedCount = groups.reduce((total, { options, value }) => total + options.filter((option) => value.includes(option)).length, 0);
  const allSelected = totalCount > 0 && selectedCount === totalCount;
  return (
    <Box sx={{ borderTop: 1, borderColor: 'dashboard.border', pt: '14px', ml: '6px' }}>
      <Typography variant="overline" sx={{ display: 'block', ...textStyles.captionCompactEmphasized, color: 'text.secondary', pb: '10px', mb: '2px' }}>Filters</Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '6px', height: '28px', padding: '2px 0 10px', ...textStyles.captionCompact, color: 'text.secondary' }}>
        <span>{selectedCount} of {totalCount} selected</span>
        <ButtonBase disabled={disabled || !totalCount}
          onClick={() => groups.forEach(({ options, onChange }) => onChange(allSelected ? [] : [...options]))}
          sx={{ ml: 'auto', ...textStyles.captionCompactEmphasized, whiteSpace: 'nowrap', color: 'dashboard.sidebarAccentText', '&.Mui-disabled': { color: 'text.muted' } }}>
          {allSelected ? 'Deselect all' : 'Select all'}
        </ButtonBase>
      </Box>

      <Box sx={{ display: { xs: 'block', sm: 'grid', md: 'block' }, gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', columnGap: '14px' }}>
        <FilterGroup label="Targets" kind="target" options={data.targets ?? []} value={state.targets ?? []} onChange={state.setTargets} disabled={disabled} />
        <FilterGroup label="Suites" kind="suite" options={data.suites ?? []} value={state.suites ?? []} onChange={state.setSuites} disabled={disabled} />
        <FilterGroup label="Thread policy" kind="mode" options={data.modes ?? []} value={state.modes ?? []} onChange={state.setModes} disabled={disabled} />
      </Box>
    </Box>
  );
}
