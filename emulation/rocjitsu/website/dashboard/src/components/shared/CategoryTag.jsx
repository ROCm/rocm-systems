import { Box, Chip } from '@mui/material';
import { alpha } from '@mui/material/styles';
import { categoryColor, sansFont } from '../../theme/tokens';

const shapes = { target: 'square', suite: 'circle', mode: 'triangle' };

// Shared nonverbal identity: labels always remain visible and accessible.
export function CategoryMarker({ label, kind = 'suite', sx }) {
  const shape = shapes[kind] ?? 'circle';
  return <Box component="span" aria-hidden="true" data-category-shape={shape} sx={[{
    display: 'inline-block', width: shape === 'triangle' ? 9 : 7, height: shape === 'triangle' ? 9 : 7,
    flexShrink: 0,
    bgcolor: (theme) => categoryColor(theme.palette.mode, label, theme.palette.text.secondary),
    borderRadius: shape === 'circle' ? '50%' : shape === 'square' ? '1px' : 0,
    ...(shape === 'triangle' ? { clipPath: 'polygon(50% 0, 100% 100%, 0 100%)' } : {}),
  }, ...(Array.isArray(sx) ? sx : [sx])]} />;
}

export default function CategoryTag({ label, kind = 'suite', sx, ...props }) {
  return <Chip {...props} label={<Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}><CategoryMarker label={label} kind={kind} /><span>{label}</span></Box>}
    data-category={kind} size="small" sx={[{
      height: 22, borderRadius: '3px', fontSize: 12, lineHeight: '18px', fontWeight: 500, fontFamily: sansFont,
      color: (theme) => categoryColor(theme.palette.mode, label, theme.palette.text.secondary),
      border: 0,
      bgcolor: (theme) => theme.palette.mode === 'dark' ? theme.palette.dashboard.subtleSurface : alpha(categoryColor('light', label, theme.palette.text.secondary), 0.07),
      '& .MuiChip-label': { px: 0.75, display: 'flex', alignItems: 'center' },
    }, ...(Array.isArray(sx) ? sx : [sx])]} />;
}
