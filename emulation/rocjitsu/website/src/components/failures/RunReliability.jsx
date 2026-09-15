import {
  Box,
  Chip,
  Divider,
  LinearProgress,
  Stack,
  Typography,
  useTheme,
} from '@mui/material';
import Chart from '../shared/Chart';
import SectionCard from '../shared/SectionCard';
import { escapeHtml, formatFullDate, formatShortDate, shortSha } from '../../utils/formatters';
import { chartAreaGradient, chartLineStyle, chartPointStyle } from '../../utils/chartStyles';

function SummaryStat({ label, value }) {
  return (
    <Box sx={{ p: 1.25, border: 1, borderColor: 'divider', borderRadius: 2, bgcolor: 'action.hover' }}>
      <Typography variant="caption" color="text.secondary">{label}</Typography>
      <Typography sx={{ fontSize: 20, fontWeight: 760, lineHeight: 1.2, mt: 0.25 }}>{value}</Typography>
    </Box>
  );
}

export default function RunReliability({ reliability }) {
  const theme = useTheme();
  const rates = reliability.rows.map((row) => row.completionPercent).filter(Number.isFinite);
  const lowestRate = Math.min(...rates, 100);
  const yAxisMinimum = Math.max(0, Math.floor((lowestRate - 2) / 5) * 5);
  const option = {
    tooltip: {
      trigger: 'axis',
      backgroundColor: theme.palette.background.paper,
      borderColor: theme.palette.divider,
      textStyle: { color: theme.palette.text.primary },
      formatter: (points) => {
        const row = reliability.rows[points[0]?.dataIndex];
        if (!row) return '';
        return [
          `<strong>${escapeHtml(formatFullDate(row.run.timestamp))}</strong>`,
          `Commit ${escapeHtml(shortSha(row.run))}`,
          `Coverage ${row.completed}/${row.total} (${Number.isFinite(row.completionPercent) ? `${row.completionPercent.toFixed(1)}%` : 'unavailable'})`,
          `Failed ${row.failed} · timed out ${row.timeout}`,
        ].join('<br/>');
      },
    },
    grid: { left: 48, right: 18, top: 20, bottom: 42 },
    xAxis: {
      type: 'category',
      data: reliability.rows.map((row) => formatShortDate(row.run.timestamp)),
      axisTick: { show: false },
      axisLine: { lineStyle: { color: theme.palette.divider } },
      axisLabel: { color: theme.palette.text.secondary, fontSize: 10, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      name: 'Coverage %',
      min: yAxisMinimum,
      max: 101,
      axisLabel: { color: theme.palette.text.secondary, fontSize: 10, formatter: '{value}%' },
      splitLine: { lineStyle: { color: theme.palette.divider, type: 'dashed' } },
    },
    series: [{
      name: 'Coverage',
      type: 'line',
      data: reliability.rows.map((row) => ({
        value: row.completionPercent,
        itemStyle: chartPointStyle(
          row.completionPercent < 100
            ? row.failed > 0 ? theme.palette.error.main : theme.palette.warning.main
            : theme.palette.primary.main,
          theme.palette.background.paper,
        ),
      })),
      symbol: 'circle',
      symbolSize: (value) => Number(value) < 100 ? 9 : 4,
      smooth: 0.12,
      lineStyle: { ...chartLineStyle(theme.palette.primary.main, 2.5), opacity: 0.82 },
      areaStyle: { color: chartAreaGradient(theme.palette.primary.main, 0.18), opacity: 1 },
      emphasis: { focus: 'series', scale: 1.6, lineStyle: { width: 3.1, opacity: 1 } },
      showSymbol: true,
      connectNulls: false,
    }],
  };

  return (
    <SectionCard title="Run Reliability" subtitle={`Coverage for the latest ${reliability.runCount} official runs under the current filters.`}>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', md: 'repeat(4, minmax(0, 1fr))' }, gap: 1 }}>
        <SummaryStat label="Case completion" value={Number.isFinite(reliability.completionPercent) ? `${reliability.completionPercent.toFixed(1)}%` : '—'} />
        <SummaryStat label="Complete runs" value={`${reliability.fullyCompleteRuns}/${reliability.runCount}`} />
        <SummaryStat label="Failed cases" value={reliability.failed} />
        <SummaryStat label="Timed out" value={reliability.timeout} />
      </Box>
      <Box sx={{ mx: -0.75, mt: 1 }}>
        <Chart option={option} height={230} ariaLabel="Run reliability coverage trend" />
      </Box>
      <Divider sx={{ my: 1.5 }} />
      <Typography variant="overline" color="text.secondary">Runs below 100%</Typography>
      {reliability.issueRuns.length === 0 ? (
        <Typography variant="body2" color="success.main" sx={{ mt: 1 }}>All official runs in this window reached complete coverage.</Typography>
      ) : (
        <Stack sx={{ mt: 0.75, gap: 1 }}>
          {reliability.issueRuns.map((row) => (
            <Box
              key={row.run.runId}
              data-testid="reliability-issue-row"
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: 'minmax(190px, 1fr) minmax(220px, 1.5fr) 176px' },
                alignItems: 'center',
                gap: 1.25,
                p: 1.25,
                border: 1,
                borderColor: 'divider',
                borderRadius: 2,
              }}
            >
              <Box>
                <Typography variant="body2" fontWeight={700}>{formatFullDate(row.run.timestamp)}</Typography>
                <Typography variant="caption" color="primary.main" component="code">{shortSha(row.run)}</Typography>
              </Box>
              <Box>
                <Stack direction="row" sx={{ justifyContent: 'space-between', gap: 1, mb: 0.45 }}>
                  <Typography variant="caption" color="text.secondary">Coverage</Typography>
                  <Typography variant="caption" fontWeight={700}>{row.completed}/{row.total} · {row.completionPercent.toFixed(1)}%</Typography>
                </Stack>
                <LinearProgress
                  data-testid="reliability-progress"
                  variant="determinate"
                  value={row.completionPercent}
                  color={row.failed > 0 ? 'error' : 'warning'}
                  sx={{ height: 5, borderRadius: 999 }}
                />
              </Box>
              <Stack direction="row" sx={{ justifyContent: { md: 'flex-end' }, flexWrap: 'wrap', gap: 0.6 }}>
                {row.failed > 0 && <Chip size="small" color="error" variant="outlined" label={`${row.failed} failed`} />}
                {row.timeout > 0 && <Chip size="small" color="warning" variant="outlined" label={`${row.timeout} timeout`} />}
              </Stack>
            </Box>
          ))}
        </Stack>
      )}
    </SectionCard>
  );
}
