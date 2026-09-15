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
  return `${formatRunTime(run.timestamp)} · ${shortSha(run)} · ${formatRunTime(commitTimestampFor(run))}`;
}

function searchableRunText(run) {
  return [
    optionLabel(run),
    `Test run ${formatRunTime(run.timestamp)}`,
    `Commit ${formatRunTime(commitTimestampFor(run))}`,
    run.runId,
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
    <Autocomplete
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
          helperText={`Search ${options.length} runs · Showing up to ${MAX_VISIBLE_OPTIONS} matches`}
        />
      )}
      slotProps={{ listbox: { sx: { maxHeight: 360 } } }}
    />
  );
}
