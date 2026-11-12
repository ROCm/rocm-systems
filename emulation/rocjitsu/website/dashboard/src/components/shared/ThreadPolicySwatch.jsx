import { Box } from '@mui/material';
import { POLICY_LINE_WIDTH, SINGLE_POLICY_DASH, SWATCH_HEIGHT, SWATCH_WIDTH } from '../../utils/chartStyles';

// One encoding for policy controls, tags, cards and chart legends.
export default function ThreadPolicySwatch({ policy, color = 'text.secondary', legend = false, muted = false }) {
  const single = policy === 'single';
  const width = legend ? 30 : SWATCH_WIDTH;
  const height = legend ? 10 : SWATCH_HEIGHT;
  const x = width / 2;
  const y = height / 2;
  return <Box component="svg" aria-hidden="true" data-thread-policy={policy} width={width} height={height} viewBox={`0 0 ${width} ${height}`} sx={{ color, flexShrink: 0, overflow: 'visible', opacity: muted ? 0.4 : 1 }}>
    <line x1="1" y1={y} x2={width - 1} y2={y} stroke="currentColor" strokeWidth={POLICY_LINE_WIDTH} strokeDasharray={single ? SINGLE_POLICY_DASH.join(' ') : undefined} strokeLinecap="round" />
    {legend && (single
      ? <Box component="rect" x={x - 4} y={y - 4} width="8" height="8" transform={`rotate(45 ${x} ${y})`} sx={{ fill: (theme) => theme.palette.background.paper, stroke: 'currentColor', strokeWidth: 2 }} />
      : <Box component="circle" cx={x} cy={y} r="4" sx={{ fill: 'currentColor', stroke: (theme) => theme.palette.background.paper, strokeWidth: 2 }} />)}
  </Box>;
}
