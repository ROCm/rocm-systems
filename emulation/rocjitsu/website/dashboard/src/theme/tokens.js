// Carbon Soft palette, shared by DOM and explicit chart options.
// Category identities remain published values; color never renames a workload.
export const shapeTokens = {

  cardRadius: '12px', controlRadius: '8px',
  cardPadding: '20px', headerPadding: '24px', rowPadding: '8px',

  cardHeaderPadding: '12px',
};

export const colorTokens = {
  light: {
    canvas: '#f1f3f7', surface: '#ffffff', sidebar: '#e7ebf2', subtleSurface: '#eef1f6',
    textPrimary: '#2b3b5a', textSecondary: '#4a5b78', textMuted: '#667690',
    border: '#dde1e9', borderSubtle: '#eaedf3', controlBorder: '#8a94a6',
    interactive: '#0f62fe', interactiveText: '#0043ce', sidebarAccentText: '#3b5278', onAccent: '#ffffff',
    selection: '#e9edf5', selectionText: '#3b5278', selectionBorder: '#c6cfe0', selectionAccent: '#3b5278',
    successText: '#198038', successGraphic: '#24a148', successSurface: '#defbe6',
    errorText: '#da1e28', errorGraphic: '#da1e28', errorSurface: '#fff1f1',
    warningText: '#8a3800', warningSurface: '#fff1e6',
    purpleText: '#6929c4', purpleSurface: '#f6f2ff', hover: '#f5f7fb',
    tableHeader: '#eef1f6', tableGroup: '#f7f9fc', barTrack: '#e6eaf2', commitText: '#3a4a68',
    // Decorative identities, not success/warning/error signals.
    secondaryAccent: '#007d79', seriesSecondary: '#c2570c', seriesSecondaryText: '#a94b0a',
    goldAccent: '#a17812', roseAccent: '#ad5478', roseText: '#9e456a', oliveAccent: '#727b3d', oliveText: '#606a30',
    shadowColor: '#101828',
    categoryTagSurface: '#eef1f6',
  },
  dark: {
    canvas: '#0e1218', surface: '#171d26', sidebar: '#121820', subtleSurface: '#222a36',
    textPrimary: '#e3e9f5', textSecondary: '#b4c0d9', textMuted: '#8f9db9',
    border: '#2f3948', borderSubtle: '#232c39', controlBorder: '#7b879b',
    interactive: '#78a9ff', interactiveText: '#9dbfff', sidebarAccentText: '#a9bbd8', onAccent: '#0e1218',
    selection: '#242f45', selectionText: '#d0daec', selectionBorder: '#3f506f', selectionAccent: '#a9bbd8',
    successText: '#42be65', successGraphic: '#42be65', successSurface: '#044317',
    errorText: '#ff8389', errorGraphic: '#fa4d56', errorSurface: '#520408',
    warningText: '#ffa04d', warningSurface: '#3a2208',
    purpleText: '#be95ff', purpleSurface: '#31135e', hover: '#1e2837',
    tableHeader: '#222a36', tableGroup: '#1c2430', barTrack: '#222a36', commitText: '#b4c0d9',
    secondaryAccent: '#3ddbd9', seriesSecondary: '#ffa04d', seriesSecondaryText: '#ffa04d',
    goldAccent: '#e0b94f', roseAccent: '#e58db0', roseText: '#e58db0', oliveAccent: '#b5c06a', oliveText: '#b5c06a',
    shadowColor: '#000000',
    categoryTagSurface: '#2a3446',
  },
};

// Suites and thread policies have fixed identities; target colors follow list position: the first seven are fixed and later ones are generated (targetColors.js).
// Suites = purple/teal/rose/olive.
export const suiteColors = {
  light: { Triton: colorTokens.light.purpleText, Llama: colorTokens.light.secondaryAccent, DeepSeek: colorTokens.light.roseAccent, TensileLite: colorTokens.light.oliveAccent },
  dark: { Triton: colorTokens.dark.purpleText, Llama: colorTokens.dark.secondaryAccent, DeepSeek: colorTokens.dark.roseAccent, TensileLite: colorTokens.dark.oliveAccent },
};

// Unknown (newly published) suites hash into the suite hues, stable across series order.
export function suiteColor(mode, label) {
  if (Object.hasOwn(suiteColors[mode], label)) return suiteColors[mode][label];
  const swatches = [...new Set(Object.values(suiteColors[mode]))];
  const hash = Array.from(String(label)).reduce((value, character) => (value * 31 + character.codePointAt(0)) >>> 0, 0);
  return swatches[hash % swatches.length];
}

// Tiny identity text needs stronger contrast than its decorative marker/plot hue.
export function suiteTextColor(mode, label) {
  const color = suiteColor(mode, label);
  const palette = colorTokens[mode];
  if (color === palette.roseAccent) return palette.roseText;
  if (color === palette.oliveAccent) return palette.oliveText;
  return color;
}
