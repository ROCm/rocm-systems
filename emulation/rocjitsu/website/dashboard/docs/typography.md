# Typography

[`src/theme/typography.js`](../src/theme/typography.js) owns the approved font types, size scale and semantic text roles. Read the exported lists there rather than copying their values into this guide or components.

Current presentation uses the **native system sans throughout** (no bundled or downloaded fonts and no monospace stack), including measurements, hashes and metadata. Numeric roles use the same sans with tabular figures.

## Choose a role first

| Content | Shared role |
|---|---|
| Product wordmark only | `brand` (the sole larger-size exception) |
| Page heading | `pageTitle` / MUI `h1` |
| Dialog heading | `dialogTitle` / MUI `h6` |
| Card or section heading | `sectionTitle` / MUI `h2` |
| Subsection heading | `subsectionTitle` / MUI `h3` |
| Body text, descriptions and table values | `body` / MUI `body2` |
| Actions and emphasized labels | `label` / MUI `button` |
| Helper text and secondary information | `caption` or `captionEmphasized` |
| Dense sidebar and header labels (16px line) | `captionCompact` or `captionCompactEmphasized` |
| Table headings | `tableHeader` |
| Measurements and identifiers | `numeric` |
| Emphasized metric results | `numericEmphasized` |
| Large comparison statistics | `summaryValue` |
| Chart axes and labels | `chartLabel` |
| Informational popups | `tooltip` |

```jsx
<Typography variant="h2">Section heading</Typography>
<Typography variant="numericEmphasized">1.24×</Typography>
```

For a non-Typography component, spread the matching `textStyles` role into `sx`. Keep color in palette roles, not in font types.

## APIs that require numeric values

ECharts requires numeric font sizes. Import `fontSizes` and `fontWeights` from the same module; do not introduce local numeric sizes or weights. The family/weight combinations are applied through the `textStyles` roles; `fontTypes` is internal to `typography.js`.

## Tables

[`src/theme/tableStyles.js`](../src/theme/tableStyles.js) owns table density choices and compared-number formatting:

- `tableStyles.normal`: benchmark, metadata, comparison and catalog membership tables.
- `tableStyles.recentRuns`: fixed-height summary rows; retain the approved partial-row viewport and centered summary columns.
- `numericTableCell`: right-aligned numeric text with tabular numbers for directly compared measurements. Keep descriptive fields left-aligned; Recent runs is the intentional centered-summary exception.

Use the shared MUI `h3` variant for equivalent subsection headings, rather than selecting sizes or weights locally.

## CSS and exceptions

- Plain CSS uses the font family and weight custom properties installed by the theme; they are not a second size scale.
- Floating labels and their notches use the same caption role without fractional text scaling, so their glyph widths and side gaps match. Preserve the shared inset/gap geometry and transparent backgrounds.
- Icon dimensions, chart symbols and layout measurements are not text sizes and keep their own geometry.
- No font files are bundled or registered. `fontWeights` has three tiers: `regular` 500, `medium` 600, `emphasized` 700. The native family decides which faces exist, so the result depends on the installed faces: on one Linux setup 500 renders as Regular and 600/700 as Bold, on another Chromium picks Medium for 500 and Bold for 600 and 700, and variable system fonts render all three. Do not rely on weight alone to tell roles apart, and do not declare a local family or weight for an exception.
- Preserve proper names, identifiers and sentence case. A typography change must not rewrite publication data.
- When a new component seems to need another font size or weight, first reuse the closest existing role. Expand the central scale only through an explicit design decision.
