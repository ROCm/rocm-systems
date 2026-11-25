import { fontSizes } from '../../theme/typography';

// Shared 32px control height; width follows the text (no fixed/min widths).
// Classic segmented control: one outer border, flush buttons, 1px dividers between
// them. Every line uses the same colour and weight, and selection never changes
// font weight, so selecting cannot resize a button.
export const CONTROL_HEIGHT = 32;
// The theme gives buttons a 44px touch target below the sm breakpoint; the group must not clip it.
const TOUCH_HEIGHT = 44;
export const controlGroupSx = (theme) => {
  const line = `1px solid ${theme.palette.dashboard?.controlBorder ?? theme.palette.divider}`;
  return {
    height: CONTROL_HEIGHT, [theme.breakpoints.down('sm')]: { height: TOUCH_HEIGHT }, width: 'fit-content', boxSizing: 'border-box', overflow: 'hidden', borderRadius: '8px', border: line,
    '& .MuiToggleButton-root': {
      height: '100%', minWidth: 0, px: '12px', py: 0, m: 0, border: 0, borderRadius: 0, whiteSpace: 'nowrap',
      fontSize: fontSizes.body, fontWeight: 'inherit', textTransform: 'none',
      '&:not(:first-of-type)': { borderLeft: line },
      '&.Mui-selected': { fontWeight: 'inherit' },
      '&.Mui-focusVisible': { outline: `2px solid ${theme.palette.primary.main}`, outlineOffset: -2 },
    },
  };
};
