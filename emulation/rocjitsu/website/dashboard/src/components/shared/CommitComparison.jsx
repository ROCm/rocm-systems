import { fontWeights } from '../../theme/typography';
import { Box, Typography } from '@mui/material';
import { sansFont } from '../../theme/tokens';
import { shortSha } from '../../utils/formatters';

export default function CommitComparison({ candidate, baseline, align = 'left', sx }) {
  if (!candidate || !baseline) {
    return (
      <Typography
        component="span"
        variant="caption"
        sx={{ color: 'text.muted', display: 'block', textAlign: align, lineHeight: '18px', ...sx }}
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
      data-testid="compared-commits"
      aria-label={`Candidate commit ${candidateSha} versus baseline commit ${baselineSha}`}
      sx={{ color: 'text.secondary', display: 'block', textAlign: align, lineHeight: '18px', whiteSpace: 'nowrap', ...sx }}
    >
      <Box component="code" sx={{ fontFamily: sansFont, fontSize: 'inherit', fontWeight: fontWeights.emphasized, color: 'dashboard.commitText' }}>{candidateSha}</Box>
      <Box component="span" sx={{ mx: '4px', color: 'text.muted' }}>{' vs '}</Box>
      <Box component="code" sx={{ fontFamily: sansFont, fontSize: 'inherit', fontWeight: fontWeights.emphasized, color: 'dashboard.commitText' }}>{baselineSha}</Box>
    </Typography>
  );
}
