import {
  Autocomplete,
  Box,
  ListItemText,
  TextField,
} from '@mui/material';
import { commitTimestampFor } from '../../data/runOrdering';
import { shortSha } from '../../utils/formatters';

const MAX_VISIBLE_OPTIONS = 50;
const runTimeFormatter = new Intl.DateTimeFormat(undefined, {
  year: 'numeric',
  month: 'short',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
  timeZone: 'UTC',
});

function formatRunTime(timestamp) {
  return `${runTimeFormatter.format(new Date(timestamp))} UTC`;
}

function optionLabel(run) {
  return `${shortSha(run)} · Run ${formatRunTime(run.timestamp)}`;
}

function searchableRunText(run) {
  return [
    optionLabel(run),
    `Test run ${formatRunTime(run.timestamp)}`,
    `Commit ${formatRunTime(commitTimestampFor(run))}`,
    run.runId,
    run.branch,
    ...(run.modes ?? []),
    ...(run.targets ?? []),
    run.timestamp,
    commitTimestampFor(run),
    run.provenance?.rocjitsuCommitSha,
    run.provenance?.commitMessage,
  ].filter(Boolean).join(' ').toLowerCase();
}

function filterRunOptions(options, { inputValue }) {
  const query = inputValue.trim().toLowerCase();
  const matches = query
    ? options.filter((run) => searchableRunText(run).includes(query))
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
            <ListItemText
              primary={`Test run · ${formatRunTime(run.timestamp)}`}
              secondary={(
                <>
                  <Box component="span" sx={{ fontFamily: 'monospace', fontWeight: 700 }}>{shortSha(run)}</Box>
                  {` · Commit · ${formatRunTime(commitTimestampFor(run))}`}
                  <Box component="span" sx={{ display: 'block' }}>{run.branch} · {(run.modes ?? []).join(' / ')} · {run.runId}</Box>
                </>
              )}
              slotProps={{
                primary: { variant: 'body2', fontWeight: 680 },
                secondary: { component: 'div', variant: 'caption', sx: { mt: 0.2 } },
              }}
            />
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
        popper: { sx: { maxWidth: 'calc(100vw - 24px)' } },
      }}
    />
    {value && <Box
      data-testid={`${label.toLowerCase().replaceAll(' ', '-')}-selected-identity`}
      sx={{ mt: 0.75, px: 0.5, fontFamily: 'monospace', fontSize: 11, color: 'text.secondary', overflowWrap: 'anywhere' }}
    >{value.runId}<Box sx={{ mt: 0.25, fontFamily: 'inherit' }}>{value.branch} · {(value.modes ?? []).join(' / ')}</Box></Box>}
    </Box>
  );
}
