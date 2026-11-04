import { fontSizes, fontWeights } from '../../theme/typography';
import { Box, Chip } from '@mui/material';
import { alpha } from '@mui/material/styles';
import { colorTokens } from '../../theme/tokens';
import ThreadPolicySwatch from './ThreadPolicySwatch';
import { softenColor } from '../../utils/colorMath';
import { markerColor, textColor } from './tagColors';

const shapes = { target: 'square', suite: 'circle', mode: 'triangle' };

// Shared nonverbal identity: labels always remain visible and accessible.
export function CategoryMarker({ label, kind = 'suite', size, muted = false }) {
  if (kind === 'mode') return <ThreadPolicySwatch policy={label} muted={muted} />;
  const shape = shapes[kind] ?? 'circle';
  const dimension = size ?? (shape === 'triangle' ? 9 : 7);
  return <Box component="span" aria-hidden="true" data-category-shape={shape} sx={[{
    display: 'inline-block', width: dimension, height: dimension,
    flexShrink: 0,
    opacity: muted ? 0.4 : 1,
    bgcolor: (theme) => textColor(theme, label, kind),
    borderRadius: shape === 'circle' ? '50%' : shape === 'square' ? '1px' : 0,
    ...(shape === 'triangle' ? { clipPath: 'polygon(50% 0, 100% 100%, 0 100%)' } : {}),
  }]} />;
}

// subdued: the medium weight and the marker's hue at 40% saturation, for dense lists.
export default function CategoryTag({ label, kind = 'suite', subdued = false, sx, ...props }) {
  return <Chip {...props} label={<Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}><CategoryMarker label={label} kind={kind} /><span>{label}</span></Box>}
    data-category={kind} size="small" sx={[{
      height: 'auto', minHeight: 20, borderRadius: '3px', lineHeight: '18px', fontSize: fontSizes.caption,
      ...(subdued && { fontWeight: fontWeights.medium }),
      color: (theme) => {
        if (kind === 'mode') return theme.palette.text.secondary;
        const text = textColor(theme, label, kind);
        return subdued ? softenColor(text, 0.4, theme.palette.mode === 'dark' ? -0.04 : 0) : text;
      },
      bgcolor: (theme) => {
        const colors = theme.palette.dashboard ?? colorTokens[theme.palette.mode];
        // Policy tags take their tint from the shared dotted/solid swatch color.
        if (kind === 'mode') return alpha(theme.palette.text.secondary, theme.palette.mode === 'dark' ? 0.14 : 0.07);
        return theme.palette.mode === 'dark' ? colors.categoryTagSurface : alpha(markerColor(theme, label, kind), 0.07);
      },
      '& .MuiChip-label': { px: '7px', py: '1px', display: 'flex', alignItems: 'center' },
    }, ...(Array.isArray(sx) ? sx : [sx])]} />;
}
