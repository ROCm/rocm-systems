import { Box, Card, CardContent, Stack, Typography } from '@mui/material';
import TimerRoundedIcon from '@mui/icons-material/TimerRounded';
import SpeedRoundedIcon from '@mui/icons-material/SpeedRounded';
import FactCheckRoundedIcon from '@mui/icons-material/FactCheckRounded';
import HealthAndSafetyRoundedIcon from '@mui/icons-material/HealthAndSafetyRounded';
import ArrowDownwardRoundedIcon from '@mui/icons-material/ArrowDownwardRounded';
import ArrowUpwardRoundedIcon from '@mui/icons-material/ArrowUpwardRounded';
import RemoveRoundedIcon from '@mui/icons-material/RemoveRounded';
import { formatDuration, formatPercent } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';
import CommitComparison from '../shared/CommitComparison';

function MetricCard({ label, value, caption, icon, tone = 'primary', badge }) {
  const toneColor = tone === 'neutral' ? 'text.secondary' : `${tone}.main`;
  return (
    <Card sx={{ position: 'relative', overflow: 'hidden' }}>
      <Box sx={{ position: 'absolute', inset: '0 auto 0 0', width: 3, bgcolor: toneColor }} />
      <CardContent sx={{ p: 2.25, '&:last-child': { pb: 2.25 } }}>
        <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'flex-start', gap: 1 }}>
          <Box>
            <Typography variant="overline" color="text.secondary">{label}</Typography>
            <Stack direction="row" sx={{ alignItems: 'baseline', gap: 1, mt: 0.65 }}>
              <Typography sx={{ fontSize: { xs: 25, xl: 29 }, lineHeight: 1.1, fontWeight: 770, letterSpacing: '-.035em' }}>{value}</Typography>
              {badge}
            </Stack>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.9 }}>{caption}</Typography>
          </Box>
          <Box sx={{ width: 38, height: 38, borderRadius: 2.3, display: 'grid', placeItems: 'center', color: toneColor, bgcolor: 'action.hover' }}>
            {icon}
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
}

export default function MetricsGrid({ metrics, candidate, baseline }) {
  const baselineState = classifyDurationChange(metrics.durationDelta);
  const baselineTone = changeTone(baselineState);
  const healthy = metrics.failed === 0;
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }, gap: 1.5 }}>
      <MetricCard
        label="Total duration"
        value={formatDuration(metrics.duration)}
        caption="Selected tests in the overview run"
        icon={<TimerRoundedIcon />}
      />
      <MetricCard
        label="Perf change"
        value={formatPercent(metrics.durationDelta)}
        caption={(
          <>
            <Box component="span" sx={{ display: 'block' }}>Selected-test aggregate vs baseline</Box>
            <CommitComparison candidate={candidate} baseline={baseline} sx={{ mt: 0.3 }} />
          </>
        )}
        icon={<SpeedRoundedIcon />}
        tone={baselineTone}
        badge={Number.isFinite(metrics.durationDelta) && (
          <Stack component="span" direction="row" sx={{ alignItems: 'center', color: baselineTone === 'neutral' ? 'text.secondary' : `${baselineTone}.main` }}>
            {baselineState === 'faster' && <ArrowDownwardRoundedIcon sx={{ fontSize: 16 }} />}
            {baselineState === 'slower' && <ArrowUpwardRoundedIcon sx={{ fontSize: 16 }} />}
            {baselineState === 'neutral' && <RemoveRoundedIcon sx={{ fontSize: 16 }} />}
          </Stack>
        )}
      />
      <MetricCard
        label="Overview run coverage"
        value={`${metrics.completed} / ${metrics.total}`}
        caption="Selected tests completed in the overview run"
        icon={<FactCheckRoundedIcon />}
        tone={metrics.completeness === 100 ? 'success' : 'warning'}
      />
      <MetricCard
        label="Run health"
        value={healthy ? 'Healthy' : `${metrics.failed} issue${metrics.failed === 1 ? '' : 's'}`}
        caption={healthy ? 'No failures or timeouts in selected tests' : 'Review incomplete cases in selected tests'}
        icon={<HealthAndSafetyRoundedIcon />}
        tone={healthy ? 'success' : 'error'}
      />
    </Box>
  );
}
