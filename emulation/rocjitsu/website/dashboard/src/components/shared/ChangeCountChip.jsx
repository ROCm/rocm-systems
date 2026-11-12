import { fontSizes } from '../../theme/typography';
import { Chip } from '@mui/material';
import { alpha } from '@mui/material/styles';

// Outcome text, a faint tint of the same hue and a border from that hue, so fill and outline agree.
export default function ChangeCountChip({ tone = 'neutral', label }) {
  const hue = (theme) => tone === 'neutral' ? theme.palette.text.secondary : theme.palette[tone].main;
  return <Chip size="small" variant="outlined" label={label} sx={{
    color: hue, borderRadius: '4px', height: 'auto',
    bgcolor: (theme) => tone === 'neutral' ? theme.palette.dashboard.subtleSurface : alpha(hue(theme), 0.08),
    borderColor: (theme) => alpha(hue(theme), tone === 'neutral' ? 0.3 : 0.4),
    '& .MuiChip-label': { padding: '2px 9px', fontSize: fontSizes.body, lineHeight: '18px' },
  }} />;
}
