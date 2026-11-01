import { fontSizes } from '../../theme/typography';
import { Chip } from '@mui/material';

// Outcome text and soft surfaces carry meaning without competing outline colors.
export default function ChangeCountChip({ tone = 'neutral', label }) {
  const color = tone === 'neutral' ? 'text.secondary' : `${tone}.main`;
  const bgcolor = tone === 'neutral' ? 'dashboard.subtleSurface' : `dashboard.${tone}Surface`;
  return <Chip size="small" variant="outlined" label={label} sx={{ color, bgcolor, borderColor: 'dashboard.border', borderRadius: '4px', height: 'auto', '& .MuiChip-label': { padding: '2px 9px', fontSize: fontSizes.body, lineHeight: '18px' } }} />;
}
