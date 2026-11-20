import { fontSizes } from '../../theme/typography';
import { Box, Paper, Stack, Typography, useTheme } from '@mui/material';
import { alpha } from '@mui/material/styles';
import SpeedRoundedIcon from '@mui/icons-material/SpeedRounded';
import TimerRoundedIcon from '@mui/icons-material/TimerRounded';
import HistoryRoundedIcon from '@mui/icons-material/HistoryRounded';
import FactCheckRoundedIcon from '@mui/icons-material/FactCheckRounded';
import FilterAltOffRoundedIcon from '@mui/icons-material/FilterAltOffRounded';
import ArrowDownwardRoundedIcon from '@mui/icons-material/ArrowDownwardRounded';
import ArrowUpwardRoundedIcon from '@mui/icons-material/ArrowUpwardRounded';
import { colorTokens, shapeTokens } from '../../theme/tokens';
import { formatDuration, formatPercent } from '../../utils/formatters';
import { changeTone, classifyDurationChange, DURATION_CHANGE_BAND_PERCENT } from '../../utils/performance';


function Stat({ label, value, icon: Icon, tone = 'neutral', accent, unavailable = false, caption, children }) {
  const theme = useTheme();
  const colors = theme.palette.dashboard ?? colorTokens[theme.palette.mode];
  const muted = unavailable || value === '—';
  const color = muted ? colors.textMuted : tone === 'success' || tone === 'error' ? theme.palette[tone].main : theme.palette.text.primary;
  // Outcome color belongs to the aggregate; small icon accents identify other metrics.
  const iconColor = muted ? colors.textMuted : accent ? colors[accent] : color;
  return <Box data-testid={`comparison-metric-${label.toLowerCase().replaceAll(' ', '-')}`} data-tone={tone} sx={{ position: 'relative', minWidth: 0, padding: '14px 16px', borderRight: 1, borderBottom: { xs: 1, lg: 0 }, borderColor: 'dashboard.border' }}>
    <Typography variant="overline" sx={{ display: 'block', color: 'text.secondary', fontSize: fontSizes.caption, lineHeight: '18px', pr: '36px' }}>{label}</Typography>
    <Box aria-hidden="true" sx={{ position: 'absolute', top: 16, right: 16, width: 32, height: 32, borderRadius: '7px', display: 'grid', placeItems: 'center', color: iconColor, bgcolor: alpha(iconColor, 0.1) }}><Icon sx={{ fontSize: 20 }} /></Box>
    <Typography component="div" variant="summaryValue" sx={{ mt: '4px', color, overflowWrap: 'anywhere' }}>{value}</Typography>
    <Typography component="div" variant="caption" sx={{ mt: '4px', lineHeight: '18px', color: 'text.muted' }}>{caption}</Typography>
    {children}
  </Box>;
}

export default function ComparisonMetrics({ model, candidate, baseline }) {
  const tolerance = model.tolerance ?? DURATION_CHANGE_BAND_PERCENT;
  const hasPair = Boolean(candidate && baseline);
  const matched = model.comparable.length;
  const state = classifyDurationChange(model.aggregateDelta, tolerance);
  const tone = changeTone(state);
  const Arrow = state === 'faster' ? ArrowDownwardRoundedIcon : state === 'slower' ? ArrowUpwardRoundedIcon : null;
  return <Paper variant="outlined" data-testid="comparison-metric-strip" sx={{ display: 'grid', borderRadius: shapeTokens.controlRadius, overflow: 'hidden', gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', sm: 'repeat(3, minmax(0, 1fr))', lg: 'repeat(5, minmax(0, 1fr))' }, '& > :first-of-type': { gridColumn: { xs: '1 / -1', sm: 'auto' } }, '& > :last-child': { borderRight: 0 } }}>
    <Stat label="Aggregate change" icon={SpeedRoundedIcon} tone={tone} unavailable={!Number.isFinite(model.aggregateDelta)} value={<Stack direction="row" sx={{ alignItems: 'center', gap: '2px' }}>{Arrow && <Arrow sx={{ fontSize: 18, flexShrink: 0 }} />}<Box component="span" data-metric-number sx={{ minWidth: 0, overflowWrap: 'anywhere' }}>{formatPercent(model.aggregateDelta)}</Box></Stack>} caption={Number.isFinite(model.aggregateDelta) ? state === 'neutral' ? `Within ±${tolerance}% noise tolerance` : state === 'faster' ? 'Less matched benchmark time' : 'More matched benchmark time' : 'No percentage comparison available'}>

    </Stat>
    <Stat label="Candidate total" icon={TimerRoundedIcon} accent="secondaryAccent" value={matched ? formatDuration(model.candidateDuration) : '—'} caption="Matched benchmark runtime" />
    <Stat label="Baseline total" icon={HistoryRoundedIcon} accent="secondaryAccent" value={matched ? formatDuration(model.baselineDuration) : '—'} caption="Matched benchmark runtime" />
    <Stat label="Comparable" icon={FactCheckRoundedIcon} accent="oliveAccent" value={hasPair ? matched : '—'} caption="Completed results in both runs" />
    <Stat label="Not comparable" icon={FilterAltOffRoundedIcon} tone={model.notComparable.length > 0 ? 'warning' : 'neutral'} accent={model.notComparable.length > 0 ? 'warningText' : undefined} value={hasPair ? model.notComparable.length : '—'} caption="Excluded from comparison" />
  </Paper>;
}
