import { Box, ListItemText } from '@mui/material';
import { commitTimestampFor } from '../../data/runOrdering';
import { monoFont } from '../../theme/tokens';
import { formatRunTime, shortSha } from '../../utils/formatters';

export default function RunOptionDetails({ run, result }) {
  return <ListItemText
    primary={`Test run · ${formatRunTime(run.timestamp)}`}
    secondary={<>
      <Box component="span" sx={{ fontFamily: monoFont, fontWeight: 700 }}>{shortSha(run)}</Box>
      {` · Commit · ${formatRunTime(commitTimestampFor(run))}`}
      <Box component="span" sx={{ display: 'block' }}>{run.branch} · {(run.modes ?? []).join(' / ')} · {run.runId}</Box>
      {result && <Box component="span" sx={{ display: 'block' }}>{result}</Box>}
    </>}
    slotProps={{ primary: { variant: 'body2', fontWeight: 680 }, secondary: { component: 'div', variant: 'caption', sx: { mt: 0.2 } } }}
  />;
}
