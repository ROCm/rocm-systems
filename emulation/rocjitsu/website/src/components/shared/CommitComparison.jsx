import { Box, Typography } from '@mui/material';
import { shortSha } from '../../utils/formatters';

export default function CommitComparison({ candidate, baseline, align = 'left', sx }) {
  if (!candidate || !baseline) {
    return (
      <Typography
        component="span"
        variant="caption"
        color="text.disabled"
        sx={{ display: 'block', textAlign: align, lineHeight: 1.35, ...sx }}
      >
        No comparison commits
      </Typography>
    );
  }

  const candidateSha = shortSha(candidate);
  const baselineSha = shortSha(baseline);
  return (
    <Typography
      component="span"
      variant="caption"
      color="text.secondary"
      data-testid="compared-commits"
      aria-label={`Candidate commit ${candidateSha} versus baseline commit ${baselineSha}`}
      title={`Candidate commit ${candidateSha} vs baseline commit ${baselineSha}`}
      sx={{ display: 'block', textAlign: align, lineHeight: 1.35, whiteSpace: 'nowrap', ...sx }}
    >
      <Box component="code" sx={{ fontFamily: 'monospace', fontSize: 'inherit', fontWeight: 700 }}>{candidateSha}</Box>
      <Box component="span" sx={{ mx: 0.45, color: 'text.disabled' }}>{' vs '}</Box>
      <Box component="code" sx={{ fontFamily: 'monospace', fontSize: 'inherit', fontWeight: 700 }}>{baselineSha}</Box>
    </Typography>
  );
}
