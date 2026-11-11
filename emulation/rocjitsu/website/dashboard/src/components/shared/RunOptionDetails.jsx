import { fontWeights, sansFont } from '../../theme/typography';
import { Box, ListItemText } from '@mui/material';
import { commitTimestampFor } from '../../data/runOrdering';
import { formatRunTime, shortSha } from '../../utils/formatters';

export default function RunOptionDetails({ run, result }) {
  return <ListItemText
    primary={`Test run · ${formatRunTime(run.timestamp)}`}
    secondary={<>
      <Box component="span" sx={{ fontFamily: sansFont, fontWeight: fontWeights.emphasized }}>{shortSha(run)}</Box>
      {` · Commit · ${formatRunTime(commitTimestampFor(run))}`}
      <Box component="span" sx={{ display: 'block' }}>{run.branch} · {run.runId}</Box>
      {result && <Box component="span" sx={{ display: 'block' }}>{result}</Box>}
    </>}
    slotProps={{ primary: { variant: 'body2', fontWeight: fontWeights.emphasized }, secondary: { component: 'div', variant: 'caption', sx: { mt: '2px' } } }}
  />;
}
