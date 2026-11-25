import { alpha } from '@mui/material/styles';
import { textStyles } from './typography';

// Two table types: `normal` (content height) and `recentRuns` (fixed row height).
// Both live in ScrollTableContainer, which owns the outer border and corners.
// Deliberate density choices; avoid component-local padding variants.
export const tableDensities = Object.freeze({
  normal: Object.freeze({ padding: '11px 12px' }),
  recentRuns: Object.freeze({ padding: '8px 12px' }),
});

export const tableStyles = Object.freeze({
  // The container draws the outer border, so the last row needs no rule of its own.
  normal: { '& td, & th': tableDensities.normal, '& tbody tr:last-child > td, & tbody tr:last-child > th': { borderBottom: 0 } },
  recentRuns: {
    '& td, & th': { ...tableDensities.recentRuns, boxSizing: 'border-box' },
    '& thead th': { height: 44, whiteSpace: 'nowrap' },
    '& tbody td': { height: 60, verticalAlign: 'middle', whiteSpace: 'nowrap' },
  },
});

// Apply only to measured values, not descriptive fields or metadata/code.
export const numericTableCell = Object.freeze({ ...textStyles.numeric, textAlign: 'right' });

// Rows whose values differ between the compared runs: a very light blue fill (6% of the primary
// blue, more saturated than the blue-grey header/group rows) plus a 3px blue bar on the first cell.
export const changedRowSx = {
  '& > .MuiTableCell-root': { bgcolor: (theme) => alpha(theme.palette.primary.main, 0.06) },
  '& > :first-child': { boxShadow: (theme) => `inset 3px 0 0 ${theme.palette.primary.main}` },
};
