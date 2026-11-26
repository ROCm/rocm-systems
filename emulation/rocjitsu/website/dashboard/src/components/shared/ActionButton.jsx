import { Button } from '@mui/material';
import { alpha } from '@mui/material/styles';
import { fontWeights } from '../../theme/typography';

// The one blue action style: bold blue text with a soft underline that firms up on hover.
// Neutral Cancel/Close, Swap and alert actions keep their own colors.
export default function ActionButton({ sx, ...props }) {
  return <Button {...props} variant="text" color="primary" disableRipple sx={[
    (theme) => ({
      whiteSpace: 'nowrap', color: 'primary.dark', fontWeight: fontWeights.emphasized,
      borderRadius: 0, px: '2px', minWidth: 0,
      // A real text underline hugs the label (and skips the icon) whatever the button's height.
      textDecoration: 'underline', textDecorationThickness: '2px', textUnderlineOffset: '4px',
      textDecorationColor: alpha(theme.palette.primary.main, 0.35),
      '&:hover': { bgcolor: 'transparent', textDecorationLine: 'underline', textDecorationColor: theme.palette.primary.main },
      '&.Mui-disabled': { textDecoration: 'none' },
    }),
    ...(Array.isArray(sx) ? sx : [sx]),
  ]} />;
}
