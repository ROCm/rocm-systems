import { alpha, Box, Checkbox, FormControlLabel, Typography } from '@mui/material';
import { CategoryMarker } from '../shared/CategoryTag';

function FilterGroup({ label, options, value, onChange, disabled, kind }) {
  return (
    <Box role="group" aria-label={label} data-testid={`${label.toLowerCase().replaceAll(' ', '-')}-filter`} sx={{ borderTop: (theme) => `1px solid ${alpha(theme.palette.text.primary, 0.3)}`, m: 0, px: 1, py: 2.25, minWidth: 0 }}>
      <Typography variant="overline" sx={{ display: 'block', mb: 1, color: 'text.secondary' }}>{label}</Typography>
      {options.map((option) => (
        <FormControlLabel key={option} sx={{ m: 0, display: 'flex', minHeight: { xs: 44, md: 36 }, borderRadius: '5px', '&:hover': { bgcolor: disabled ? 'transparent' : 'action.hover' } }}
          control={<Checkbox size="small" disabled={disabled} checked={value.includes(option)}
            onChange={() => onChange(value.includes(option) ? value.filter((item) => item !== option) : [...value, option])}
            sx={{ p: 0.5, mr: 0.75, color: 'dashboard.controlBorder', '& svg': { fontSize: 20 } }} />}
          label={<Box component="span" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}><CategoryMarker label={option} kind={kind} /><Typography component="span" sx={{ fontSize: 14, lineHeight: '20px', fontWeight: 500, color: disabled ? 'text.disabled' : 'text.primary' }}>{option}</Typography></Box>} />
      ))}
      {!options.length && <Typography variant="caption" color="text.secondary">No available {label.toLowerCase()}</Typography>}
      {kind === 'mode' && <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1.25, fontSize: 11 }}>ST · single-threaded<br />MT · multi-threaded</Typography>}
    </Box>
  );
}

export default function FiltersBar({ data, state, disabled = false }) {
  return (
    <Box sx={{ mt: 2, display: { xs: 'block', sm: 'grid', md: 'block' }, gridTemplateColumns: 'repeat(3, minmax(0, 1fr))' }}>
      <FilterGroup label="Targets" kind="target" options={data.targets ?? []} value={state.targets ?? []} onChange={state.setTargets} disabled={disabled} />
      <FilterGroup label="Suites" kind="suite" options={data.suites ?? []} value={state.suites ?? []} onChange={state.setSuites} disabled={disabled} />
      <FilterGroup label="Execution modes" kind="mode" options={data.modes ?? []} value={state.modes ?? []} onChange={state.setModes} disabled={disabled} />
    </Box>
  );
}
