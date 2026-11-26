import { fontSizes } from '../../theme/typography';
import { Box, Paper, Typography } from '@mui/material';
import TimerRoundedIcon from '@mui/icons-material/TimerRounded';
import SpeedRoundedIcon from '@mui/icons-material/SpeedRounded';
import FactCheckRoundedIcon from '@mui/icons-material/FactCheckRounded';
import ChecklistRoundedIcon from '@mui/icons-material/ChecklistRounded';
import { shapeTokens } from '../../theme/tokens';
import { formatDuration } from '../../utils/formatters';
import { targetColor } from '../../theme/targetColors';
import ThreadPolicySwatch from '../shared/ThreadPolicySwatch';

function Metric({ label, rows, valueFor, caption, icon: Icon, iconColor }) {
  return <Paper variant="outlined" data-testid={`metric-card-${label.toLowerCase().replaceAll(' ', '-')}`} data-tone="neutral" sx={{ p: shapeTokens.cardPadding, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, mb: 1 }}>
      <Typography component="h3" variant="h2">{label}</Typography>
      <Icon role="img" aria-label={caption} sx={{ color: rows.some((row) => row.cohortIds.length && valueFor(row) != null) ? iconColor : 'text.muted', fontSize: 19, width: 19, height: 19, flexShrink: 0 }} />
    </Box>
    <Box sx={{ flex: 1 }}>
      {!rows.length ? <Typography color="text.secondary" sx={{ fontSize: fontSizes.body, py: '10px' }}>No configurations selected</Typography> : rows.map((row) => {
        const value = row.cohortIds.length ? valueFor(row) : null;
        const unavailable = value == null;
        const displayValue = row.cohortIds.length ? value ?? 'Unavailable' : 'No shared workloads';
        return <Box key={row.key} data-configuration={row.key} sx={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto', alignItems: 'center', gap: 1, py: shapeTokens.rowPadding, '& + &': { borderTop: 1, borderColor: 'divider' } }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: '7px', minWidth: 0 }}><ThreadPolicySwatch policy={row.mode} color={(theme) => targetColor(theme, row.target)} /><Typography variant="caption" sx={{ overflowWrap: 'anywhere', color: 'text.secondary', fontSize: fontSizes.body, lineHeight: '18px' }}>{row.target} {row.mode}</Typography></Box>
          <Typography variant="numericEmphasized" sx={{ fontSize: unavailable ? fontSizes.caption : undefined, textAlign: 'right', color: unavailable ? 'text.muted' : 'text.secondary', fontStyle: unavailable ? 'italic' : 'normal' }}>{displayValue}</Typography>
        </Box>;
      })}
    </Box>
  </Paper>;
}

export default function MetricsGrid({ projection }) {
  const rows = projection.rows;
  return <Box data-testid="metric-strip" sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', sm: 'repeat(2,minmax(0,1fr))', md: 'repeat(4,minmax(0,1fr))' } }}>
    <Metric label="Latest duration" icon={TimerRoundedIcon} iconColor="dashboard.secondaryAccent" rows={rows} valueFor={(r) => r.latest.durationSeconds == null ? null : formatDuration(r.latest.durationSeconds)} caption={projection.endpointLabel} />
    <Metric label="Relative speedup" icon={SpeedRoundedIcon} iconColor="dashboard.goldAccent" rows={rows} valueFor={(r) => r.latest.speedup == null ? null : `${r.latest.speedup.toFixed(2)}×`} caption="First valid reference / latest · higher is faster" />
    <Metric label="Latest results" icon={ChecklistRoundedIcon} iconColor="dashboard.roseAccent" rows={rows} valueFor={(r) => `${r.latest.completed}/${r.latest.required}`} caption="Succeeded / required workloads" />
    <Metric label="Valid data points" icon={FactCheckRoundedIcon} iconColor="dashboard.oliveAccent" rows={rows} valueFor={(r) => r.points.filter((point) => Number.isFinite(point.speedup)).length} caption="Plotted points in selected range" />
  </Box>;
}
