import { Box, Paper, Stack, Typography, useTheme } from '@mui/material';
import { alpha } from '@mui/material/styles';
import SpeedRoundedIcon from '@mui/icons-material/SpeedRounded';
import TimerRoundedIcon from '@mui/icons-material/TimerRounded';
import HistoryRoundedIcon from '@mui/icons-material/HistoryRounded';
import FactCheckRoundedIcon from '@mui/icons-material/FactCheckRounded';
import FilterAltOffRoundedIcon from '@mui/icons-material/FilterAltOffRounded';
import ArrowDownwardRoundedIcon from '@mui/icons-material/ArrowDownwardRounded';
import ArrowUpwardRoundedIcon from '@mui/icons-material/ArrowUpwardRounded';
import { monoFont } from '../../theme/tokens';
import { formatDuration, formatPercent } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';
import CommitComparison from '../shared/CommitComparison';

function Stat({ label, value, icon: Icon, tone = 'neutral', caption, children }) {
  const theme = useTheme();
  const color = tone === 'purple' ? (theme.palette.mode === 'dark' ? '#be95ff' : '#6929c4') : tone === 'neutral' ? theme.palette.text.secondary : theme.palette[tone].main;
  return <Box data-testid={`comparison-metric-${label.toLowerCase().replaceAll(' ', '-')}`} data-tone={tone} sx={{ position: 'relative', minWidth: 0, p: 2, borderRight: 1, borderBottom: 1, borderColor: 'divider' }}>
    <Typography variant="overline" sx={{ display: 'block', color: 'text.secondary', fontSize: 11, lineHeight: '18px', pr: 4.5 }}>{label}</Typography>
    <Box aria-hidden="true" sx={{ position: 'absolute', top: 16, right: 16, width: 32, height: 32, borderRadius: '7px', display: 'grid', placeItems: 'center', color, bgcolor: alpha(color, 0.1) }}><Icon sx={{ fontSize: 20 }} /></Box>
    <Typography component="div" sx={{ mt: 0.75, fontFamily: monoFont, fontSize: { xs: 20, sm: 24 }, lineHeight: '32px', fontWeight: 500, color, fontVariantNumeric: 'tabular-nums' }}>{value}</Typography>
    <Typography component="div" variant="caption" sx={{ mt: 0.5, fontSize: 11, lineHeight: '18px', color: 'text.secondary' }}>{caption}</Typography>
    {children}
  </Box>;
}

export default function ComparisonMetrics({ model, candidate, baseline, tolerance = 3 }) {
  const hasPair = Boolean(candidate && baseline);
  const matched = model.comparable.length;
  const state = classifyDurationChange(model.aggregateDelta, tolerance);
  const tone = changeTone(state);
  const Arrow = state === 'faster' ? ArrowDownwardRoundedIcon : state === 'slower' ? ArrowUpwardRoundedIcon : null;
  return <Paper variant="outlined" data-testid="comparison-metric-strip" sx={{ display: 'grid', borderRadius: '8px', overflow: 'hidden', boxShadow: 'none', gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', sm: 'repeat(3, minmax(0, 1fr))', lg: 'repeat(5, minmax(0, 1fr))' }, '& > :first-of-type': { gridColumn: { xs: '1 / -1', sm: 'auto' } }, '& > :last-child': { borderRight: 0 } }}>
    <Stat label="Aggregate change" icon={SpeedRoundedIcon} tone={tone} value={<Stack direction="row" sx={{ alignItems: 'center', gap: 0.25 }}>{Arrow && <Arrow sx={{ fontSize: 18 }} />}{formatPercent(model.aggregateDelta)}</Stack>} caption={Number.isFinite(model.aggregateDelta) ? state === 'neutral' ? `Within ±${tolerance}% noise tolerance` : state === 'faster' ? 'Less matched benchmark time' : 'More matched benchmark time' : 'No percentage comparison available'}>
      {hasPair && <CommitComparison candidate={candidate} baseline={baseline} sx={{ mt: 0.5, fontSize: 11, whiteSpace: 'normal', overflowWrap: 'anywhere' }} />}
    </Stat>
    <Stat label="Candidate total" icon={TimerRoundedIcon} tone="primary" value={matched ? formatDuration(model.candidateDuration) : '—'} caption="Matched benchmark runtime" />
    <Stat label="Baseline total" icon={HistoryRoundedIcon} tone="purple" value={matched ? formatDuration(model.baselineDuration) : '—'} caption="Matched benchmark runtime" />
    <Stat label="Comparable" icon={FactCheckRoundedIcon} tone={matched > 0 ? 'success' : 'neutral'} value={hasPair ? matched : '—'} caption="Completed results in both runs" />
    <Stat label="Not comparable" icon={FilterAltOffRoundedIcon} tone={model.notComparable.length > 0 ? 'warning' : 'neutral'} value={hasPair ? model.notComparable.length : '—'} caption="Excluded from comparison" />
  </Paper>;
}
