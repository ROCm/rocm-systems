import { Box, Paper, Typography } from '@mui/material';
import TimerRoundedIcon from '@mui/icons-material/TimerRounded';
import SpeedRoundedIcon from '@mui/icons-material/SpeedRounded';
import FactCheckRoundedIcon from '@mui/icons-material/FactCheckRounded';
import HealthAndSafetyRoundedIcon from '@mui/icons-material/HealthAndSafetyRounded';
import { alpha } from '@mui/material/styles';
import { monoFont } from '../../theme/tokens';
import { formatDuration, formatPercent } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';

function Metric({ label, value, caption, icon: Icon, tone = 'neutral', badgeTone = tone }) {
  return (
    <Box data-testid={`metric-card-${label.toLowerCase().replaceAll(' ', '-')}`} data-tone={tone} data-icon-tone={badgeTone} sx={{ position: 'relative', minWidth: 0, p: 2 }}>
      <Typography variant="overline" sx={{ color: 'text.secondary', fontSize: 11, lineHeight: '18px', display: 'block', pr: 4.5 }}>{label}</Typography>
      <Box aria-hidden="true" sx={{ position: 'absolute', top: 16, right: 16, width: 32, height: 32, borderRadius: '7px', display: 'grid', placeItems: 'center', color: badgeTone === 'neutral' ? 'text.secondary' : `${badgeTone}.main`, bgcolor: (theme) => alpha(badgeTone === 'neutral' ? theme.palette.text.secondary : theme.palette[badgeTone].main, 0.1) }}><Icon sx={{ fontSize: 20 }} /></Box>
      <Typography sx={{ mt: 0.75, fontSize: { xs: 20, sm: 24 }, lineHeight: '32px', fontWeight: 500, fontFamily: monoFont, fontVariantNumeric: 'tabular-nums', color: tone === 'neutral' ? 'text.primary' : `${tone}.main` }}>{value}</Typography>
      <Typography data-testid="metric-caption" variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary', lineHeight: '18px', fontSize: 11 }}>{caption}</Typography>
    </Box>
  );
}

export default function MetricsGrid({ metrics, range = 'ALL' }) {
  const hasResults = metrics.total > 0;
  const healthy = hasResults && metrics.completed === metrics.total && metrics.failed === 0;
  const period = { '1W': 'past week', '1M': 'past month', '3M': 'past 3 months', ALL: 'all available history' }[range] ?? 'selected range';
  return (
    <Paper variant="outlined" data-testid="metric-strip" sx={{ borderRadius: '8px', display: 'grid', gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', md: 'repeat(4, minmax(0, 1fr))' }, overflow: 'hidden', boxShadow: 'none', '& > div': { borderRight: 1, borderColor: 'divider' }, '& > div:nth-of-type(2)': { borderRightWidth: { xs: 0, md: 1 } }, '& > div:nth-of-type(n+3)': { borderTopStyle: 'solid', borderTopWidth: { xs: 1, md: 0 }, borderColor: 'divider' }, '& > div:last-child': { borderRight: 0 } }}>
      <Metric icon={TimerRoundedIcon} badgeTone={hasResults ? 'primary' : 'neutral'} label="Total duration" value={formatDuration(metrics.duration)} caption="Sum of selected benchmark runtimes" />
      <Metric icon={SpeedRoundedIcon} label="Perf change" value={formatPercent(metrics.durationDelta)} tone={hasResults ? changeTone(classifyDurationChange(metrics.durationDelta)) : 'neutral'} caption={`Selected period · ${period} · first vs latest commit`} />
      <Metric icon={FactCheckRoundedIcon} label="Run coverage" value={hasResults ? `${metrics.completed} / ${metrics.total}` : '—'} tone={hasResults && metrics.completed === metrics.total ? 'success' : 'neutral'} caption="Completed selected benchmark results" />
      <Metric icon={HealthAndSafetyRoundedIcon} label="Run health" value={!hasResults ? '—' : healthy ? 'OK' : metrics.failed ? 'Failed' : 'Incomplete'} tone={!hasResults ? 'neutral' : healthy ? 'success' : metrics.failed ? 'error' : 'neutral'} caption={!hasResults ? 'No selected results available' : healthy ? 'All selected tests completed' : `${metrics.total - metrics.completed} incomplete · ${metrics.failed} failed / timed out`} />
    </Paper>
  );
}
