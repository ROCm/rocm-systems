import { fontWeights, sansFont } from '../../theme/typography';
import { Box, Typography } from '@mui/material';
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
  const branchName = (run) => <Box component="span" sx={{ color: 'text.muted' }}>{` (${run.branch})`}</Box>;
  return (
    <Typography
      component="span"
      variant="caption"
      data-testid="compared-commits"
      aria-label={`Candidate commit ${candidateSha} on ${candidate.branch} versus baseline commit ${baselineSha} on ${baseline.branch}`}
      sx={{ color: 'text.secondary', display: 'block', textAlign: align, lineHeight: '18px', overflowWrap: 'anywhere', ...sx }}
    >
      <Box component="code" sx={{ fontFamily: sansFont, fontSize: 'inherit', fontWeight: fontWeights.emphasized, color: 'dashboard.commitText' }}>{candidateSha}</Box>{branchName(candidate)}
      <Box component="span" sx={{ mx: '4px', color: 'text.muted' }}>{' vs '}</Box>
      <Box component="code" sx={{ fontFamily: sansFont, fontSize: 'inherit', fontWeight: fontWeights.emphasized, color: 'dashboard.commitText' }}>{baselineSha}</Box>{branchName(baseline)}
    </Typography>
  );
}
