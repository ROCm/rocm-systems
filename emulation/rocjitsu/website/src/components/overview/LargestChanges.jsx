import {
  Box,
  Divider,
  LinearProgress,
  Stack,
  Typography,
} from '@mui/material';
import ArrowDownwardRoundedIcon from '@mui/icons-material/ArrowDownwardRounded';
import ArrowUpwardRoundedIcon from '@mui/icons-material/ArrowUpwardRounded';
import RemoveRoundedIcon from '@mui/icons-material/RemoveRounded';
import SectionCard from '../shared/SectionCard';
import { formatPercent } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';
import CommitComparison from '../shared/CommitComparison';

const NOISE_TOLERANCE = 3;
const legendItems = [
  { state: 'slower', label: 'Slower', color: 'error.main' },
  { state: 'neutral', label: `Within ±${NOISE_TOLERANCE}%`, color: 'text.disabled' },
  { state: 'faster', label: 'Faster', color: 'success.main' },
];

export default function LargestChanges({ changes, candidate, baseline }) {
  const maxDelta = Math.max(...changes.map((item) => Math.abs(item.delta)), 1);

  return (
    <SectionCard
      title="Largest Changes"
      subtitle="Largest benchmark changes in the Overview comparison"
      sx={{ height: '100%' }}
      data-testid="largest-changes"
    >
      <Stack direction="row" sx={{ flexWrap: 'wrap', columnGap: 2, rowGap: 0.75, mb: 1 }}>
        {legendItems.map((item) => (
          <Stack key={item.state} direction="row" data-legend-state={item.state} sx={{ alignItems: 'center', gap: 0.65 }}>
            <Box aria-hidden="true" sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: item.color, flexShrink: 0 }} />
            <Typography variant="caption" color="text.secondary" fontWeight={650}>{item.label}</Typography>
          </Stack>
        ))}
      </Stack>
      {changes.length === 0 ? (
        <Typography color="text.secondary" sx={{ py: 8, textAlign: 'center' }}>No comparable results in this selection.</Typography>
      ) : changes.map((item, index) => {
        const changeState = classifyDurationChange(item.delta, NOISE_TOLERANCE);
        const tone = changeTone(changeState);
        const toneColor = tone === 'neutral' ? 'text.secondary' : `${tone}.main`;
        return (
          <Box key={item.test.testId} data-change-state={changeState}>
            {index > 0 && <Divider />}
            <Box sx={{ py: 1.35 }}>
              <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', gap: 1.5 }}>
                <Box sx={{ minWidth: 0 }}>
                  <Typography variant="body2" fontWeight={680} noWrap>{item.test.name}</Typography>
                  <Typography variant="caption" color="text.secondary">{item.test.target} · {item.test.suite}</Typography>
                </Box>
                <Box sx={{ flexShrink: 0 }}>
                  <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'flex-end', color: toneColor }}>
                    {changeState === 'slower' && <ArrowUpwardRoundedIcon sx={{ fontSize: 16 }} />}
                    {changeState === 'faster' && <ArrowDownwardRoundedIcon sx={{ fontSize: 16 }} />}
                    {changeState === 'neutral' && <RemoveRoundedIcon sx={{ fontSize: 16 }} />}
                    <Typography variant="body2" fontWeight={750}>{formatPercent(item.delta)}</Typography>
                  </Stack>
                  <CommitComparison candidate={candidate} baseline={baseline} align="right" sx={{ mt: 0.15, fontSize: 10 }} />
                </Box>
              </Stack>
              <LinearProgress
                variant="determinate"
                value={(Math.abs(item.delta) / maxDelta) * 100}
                color={changeState === 'slower' ? 'error' : changeState === 'faster' ? 'success' : 'primary'}
                sx={{
                  mt: 1,
                  height: 6,
                  borderRadius: 999,
                  bgcolor: 'action.hover',
                  '& .MuiLinearProgress-bar': {
                    borderRadius: 999,
                    boxShadow: changeState === 'neutral' ? 'none' : 2,
                    ...(changeState === 'neutral' && { bgcolor: 'text.disabled' }),
                  },
                }}
              />
            </Box>
          </Box>
        );
      })}
    </SectionCard>
  );
}
