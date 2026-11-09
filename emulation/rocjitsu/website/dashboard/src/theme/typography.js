// The approved text scale. Icon dimensions are intentionally independent.
export const fontSizes = Object.freeze({ caption: 14, body: 14, tableHeader: 15, nav: 16, section: 18, dialog: 20, page: 24, brand: 32 });
// Three semantic weight tiers. The native system family decides which faces exist, so the result depends on the
// installed faces: on one Linux setup 500 renders Regular and 600/700 both render Bold; on another Chromium picked
// Medium for 500 and Bold for 600 and 700; variable system fonts (San Francisco, Segoe UI Variable) render all three.
// Roles therefore also differ by size, case and color.
export const fontWeights = Object.freeze({ regular: 500, medium: 600, emphasized: 700 });

// Native system fonts only: nothing is bundled or downloaded. system-ui resolves to the platform UI
// face (San Francisco, Segoe UI, Cantarell/Noto Sans...), the closest native match to a neo-grotesque
// UI sans; the named faces cover engines without system-ui. Every role uses this one family, numerals included.
export const sansFont = 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif';

// Family and weight form a reusable font type; color remains a palette concern.
const fontTypes = Object.freeze({
  regular: Object.freeze({ fontFamily: sansFont, fontWeight: fontWeights.regular }),
  medium: Object.freeze({ fontFamily: sansFont, fontWeight: fontWeights.medium }),
  emphasized: Object.freeze({ fontFamily: sansFont, fontWeight: fontWeights.emphasized }),
});

const role = (type, size, lineHeight) => Object.freeze({ ...fontTypes[type], fontSize: `${fontSizes[size]}px`, lineHeight: `${lineHeight}px` });
const tabular = { fontVariantNumeric: 'tabular-nums' };

// Every role uses the native sans.
// Use these roles in MUI typography/sx; use the numeric lists for chart APIs.
export const textStyles = Object.freeze({
  brand: role('emphasized', 'brand', 40),
  pageTitle: role('emphasized', 'page', 32),
  dialogTitle: role('emphasized', 'dialog', 28),
  sectionTitle: role('medium', 'section', 26),
  tableHeader: role('medium', 'tableHeader', 24),
  subsectionTitle: role('emphasized', 'body', 20),
  body: role('regular', 'body', 20),
  label: role('emphasized', 'body', 20),
  caption: role('regular', 'caption', 20),
  captionEmphasized: role('emphasized', 'caption', 20),
  // Dense sidebar/header labels: caption size and weights with a tighter line height.
  captionCompact: role('regular', 'caption', 16),
  captionCompactEmphasized: role('emphasized', 'caption', 16),
  numeric: Object.freeze({ ...role('regular', 'body', 20), ...tabular }),
  numericEmphasized: Object.freeze({ ...role('emphasized', 'body', 20), ...tabular }),
  summaryValue: Object.freeze({ ...role('emphasized', 'page', 32), ...tabular }),
  chartLabel: role('regular', 'caption', 20),
  tooltip: role('regular', 'body', 21),
});

// Plain CSS derives from the same definitions.
export const typographyCssVariables = Object.freeze({
  '--font-family-text': sansFont,
  '--font-family-numeric': sansFont,
  '--font-weight-emphasized': fontWeights.emphasized,
});
