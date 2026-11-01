import { textStyles } from './typography';

// Deliberate density choices; avoid component-local padding variants.
export const tableDensities = Object.freeze({
  normal: Object.freeze({ padding: '11px 12px' }),
  compact: Object.freeze({ padding: '6px 16px' }),
  recentRuns: Object.freeze({ padding: '8px 12px' }),
});

export const tableStyles = Object.freeze({
  normal: { '& td, & th': tableDensities.normal },
  compact: { '& td, & th': tableDensities.compact },
  recentRuns: {
    '& td, & th': { ...tableDensities.recentRuns, boxSizing: 'border-box' },
    '& thead th': { height: 44, whiteSpace: 'nowrap' },
    '& tbody td': { height: 60, verticalAlign: 'middle', whiteSpace: 'nowrap' },
  },
});

// Apply only to measured values, not descriptive fields or metadata/code.
export const numericTableCell = Object.freeze({ ...textStyles.numeric, textAlign: 'right' });
