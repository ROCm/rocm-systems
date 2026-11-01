// The approved text scale. Icon dimensions are intentionally independent.
export const fontSizes = Object.freeze({ caption: 14, body: 14, tableHeader: 15, nav: 16, section: 18, dialog: 20, page: 24, brand: 32 });
export const fontWeights = Object.freeze({ regular: 400, tableHeader: 450, medium: 500, emphasized: 600 });

export const sansFont = '"Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
export const monoFont = '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace';

// Family and weight form a reusable font type; color remains a palette concern.
export const fontTypes = Object.freeze({
  regular: Object.freeze({ fontFamily: sansFont, fontWeight: fontWeights.regular }),
  tableHeader: Object.freeze({ fontFamily: sansFont, fontWeight: fontWeights.tableHeader }),
  medium: Object.freeze({ fontFamily: sansFont, fontWeight: fontWeights.medium }),
  emphasized: Object.freeze({ fontFamily: sansFont, fontWeight: fontWeights.emphasized }),
  numeric: Object.freeze({ fontFamily: monoFont, fontWeight: fontWeights.regular }),
  numericEmphasized: Object.freeze({ fontFamily: monoFont, fontWeight: fontWeights.emphasized }),
});

const role = (type, size, lineHeight) => Object.freeze({ ...fontTypes[type], fontSize: `${fontSizes[size]}px`, lineHeight: `${lineHeight}px` });
const tabular = { fontVariantNumeric: 'tabular-nums' };

// Active roles use Sans throughout; the approved font catalog above is retained.
// Use these roles in MUI typography/sx; use the numeric lists for chart APIs.
export const textStyles = Object.freeze({
  brand: role('emphasized', 'brand', 40),
  pageTitle: role('emphasized', 'page', 32),
  dialogTitle: role('emphasized', 'dialog', 28),
  sectionTitle: role('medium', 'section', 26),
  tableHeader: role('tableHeader', 'tableHeader', 24),
  subsectionTitle: role('emphasized', 'body', 20),
  body: role('regular', 'body', 20),
  label: role('emphasized', 'body', 20),
  caption: role('regular', 'caption', 20),
  captionEmphasized: role('emphasized', 'caption', 20),
  numeric: Object.freeze({ ...role('regular', 'body', 20), ...tabular }),
  numericEmphasized: Object.freeze({ ...role('emphasized', 'body', 20), ...tabular }),
  summaryValue: Object.freeze({ ...role('emphasized', 'page', 32), ...tabular }),
  chartLabel: role('regular', 'caption', 20),
  tooltip: role('regular', 'body', 21),
});

// Plain CSS and portal-density correction derive from the same definitions.
export const typographyCssVariables = Object.freeze({
  ...Object.fromEntries(Object.entries(fontSizes).map(([name, value]) => [`--font-size-${name}`, `${value}px`])),
  '--font-family-text': sansFont,
  '--font-family-numeric': sansFont,
  '--font-weight-regular': fontWeights.regular,
  '--font-weight-emphasized': fontWeights.emphasized,
  '--line-height-body': textStyles.body.lineHeight,
  '--line-height-caption': textStyles.caption.lineHeight,
});
