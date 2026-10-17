import { Box, Stack, Typography } from '@mui/material';
import SectionCard from '../shared/SectionCard';
import CategoryTag from '../shared/CategoryTag';
import { monoFont } from '../../theme/tokens';
import { formatPercent, shortSha } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';

export default function LargestChanges({ changes, candidate, baseline }) {
  const maximumMagnitude = Math.max(...changes.map((item) => Math.abs(item.delta)).filter(Number.isFinite), 0);
  return (
    <SectionCard title="Largest Changes" subtitle="Across selected targets and suites and execution modes" sx={{ height: '100%', minWidth: 0 }} contentSx={{ p: 2, '&:last-child': { pb: 2 } }} data-testid="largest-changes">
      {changes.length === 0 ? <Typography color="text.secondary" sx={{ py: 4 }}>No comparable results in this selection.</Typography> : changes.map((item) => {
        const state = classifyDurationChange(item.delta, 3);
        const tone = changeTone(state);
        return <Box key={item.candidateTest.testId} data-change-state={state} sx={{ py: 1.1 }}>
          <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center', gap: 1 }}>
            <Box sx={{ minWidth: 0 }}>
              <Typography variant="body2" title={item.candidateTest.name} sx={{ fontWeight: 600, overflowWrap: 'anywhere', fontSize: 12 }}>{item.candidateTest.name}</Typography>
              <Stack direction="row" sx={{ mt: 0.4, gap: 0.5, flexWrap: 'wrap' }}>
                <CategoryTag kind="target" label={item.candidateTest.target} />
                <CategoryTag label={item.candidateTest.suite} />
                <CategoryTag kind="mode" label={item.candidateTest.mode} />
              </Stack>
            </Box>
            <Typography variant="body2" aria-label={`${formatPercent(item.delta)}, ${state === 'neutral' ? 'within 3% noise tolerance' : state}`} title={`Candidate ${shortSha(candidate)} vs baseline ${shortSha(baseline)}`} sx={{ flexShrink: 0, fontWeight: 650, fontFamily: monoFont, fontSize: 15, color: tone === 'neutral' ? 'text.secondary' : `${tone}.main` }}>{formatPercent(item.delta)}</Typography>
          </Stack>
          <Box aria-hidden="true" sx={{ mt: 1, height: 3, bgcolor: 'divider', borderRadius: 1 }}>
            <Box data-testid="change-magnitude-bar" sx={{ height: '100%', width: `${maximumMagnitude && Number.isFinite(item.delta) ? Math.abs(item.delta) / maximumMagnitude * 100 : 0}%`, bgcolor: tone === 'neutral' ? 'text.disabled' : `${tone}.main`, borderRadius: 1 }} />
          </Box>
        </Box>;
      })}
    </SectionCard>
  );
}
