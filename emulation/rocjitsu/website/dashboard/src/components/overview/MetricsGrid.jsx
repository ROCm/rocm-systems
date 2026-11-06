import { Box, Paper, Typography, useTheme } from '@mui/material';
import TimerRoundedIcon from '@mui/icons-material/TimerRounded';
import SpeedRoundedIcon from '@mui/icons-material/SpeedRounded';
import FactCheckRoundedIcon from '@mui/icons-material/FactCheckRounded';
import ChecklistRoundedIcon from '@mui/icons-material/ChecklistRounded';
import { monoFont, sansFont } from '../../theme/tokens';
import { formatDuration } from '../../utils/formatters';
import { configurationColor } from '../../utils/chartColors';

function Metric({ label, rows, valueFor, caption, icon: Icon, iconColor }) {
  const theme = useTheme();
  return <Paper variant="outlined" data-testid={`metric-card-${label.toLowerCase().replaceAll(' ', '-')}`} data-tone="neutral" sx={{ p: 2, minWidth: 0, borderRadius: 1, display: 'flex', flexDirection: 'column' }}>
    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, mb: 1.25 }}>
      <Typography component="h3" sx={{ color: 'text.primary', fontSize: 14, lineHeight: '24px', fontWeight: 700, letterSpacing: 0, fontFamily: `"Overview Title", ${sansFont}` }}>{label}</Typography>
      <Icon aria-hidden="true" sx={{ color: rows.some((row) => row.cohortIds.length && valueFor(row) != null) ? iconColor : 'text.secondary', fontSize: 19, flexShrink: 0 }} />
    </Box>
    <Box sx={{ flex: 1 }}>
      {!rows.length ? <Typography color="text.secondary">No configurations selected</Typography> : rows.map((row, index) => {
        const value = row.cohortIds.length ? valueFor(row) : null;
        const unavailable = value == null;
        const displayValue = row.cohortIds.length ? value ?? 'Unavailable' : 'No shared workloads';
        return <Box key={row.key} data-configuration={row.key} sx={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto', alignItems: 'baseline', gap: 1.5, py: 0 }}>
          <Typography variant="caption" sx={{ overflowWrap: 'anywhere', color: 'text.secondary', lineHeight: '18px' }}>{row.target} {row.mode}</Typography>
          <Typography title={row.latest.reason ?? undefined} sx={{ fontFamily: monoFont, fontSize: unavailable ? 12 : rows.length === 1 ? 24 : 14, lineHeight: rows.length === 1 && !unavailable ? '28px' : '18px', textAlign: 'right', fontVariantNumeric: 'tabular-nums', color: unavailable ? 'text.secondary' : configurationColor(row, index, theme.palette.mode) }}>{displayValue}</Typography>
        </Box>;
      })}
    </Box>
    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1.5, fontSize: 11, lineHeight: 1.5 }}>{caption}</Typography>
  </Paper>;
}

export default function MetricsGrid({ projection }) {
  const rows = projection.rows;
  return <Box data-testid="metric-strip" sx={{ display: 'grid', gap: 1.75, gridTemplateColumns: { xs: '1fr', sm: 'repeat(2,minmax(0,1fr))', md: 'repeat(4,minmax(0,1fr))' } }}>
    <Metric label="Latest duration" icon={TimerRoundedIcon} iconColor="#52657D" rows={rows} valueFor={(r) => r.latest.durationSeconds == null ? null : formatDuration(r.latest.durationSeconds)} caption={projection.endpointLabel} />
    <Metric label="Relative speedup" icon={SpeedRoundedIcon} iconColor="#A17812" rows={rows} valueFor={(r) => r.latest.speedup == null ? null : `${r.latest.speedup.toFixed(2)}×`} caption="First valid reference / latest · higher is faster" />
    <Metric label="Latest results" icon={ChecklistRoundedIcon} iconColor="#AD5478" rows={rows} valueFor={(r) => `${r.latest.completed}/${r.latest.required}`} caption="Succeeded / required workloads" />
    <Metric label="Valid data points" icon={FactCheckRoundedIcon} iconColor="#727B3D" rows={rows} valueFor={(r) => r.points.filter((point) => Number.isFinite(point.speedup)).length} caption="Plotted points in selected range" />
  </Box>;
}
