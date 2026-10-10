// Approved v10 Carbon-inspired palette, shared by DOM and explicit chart options.
// Category identities remain published values; color never renames a workload.
export const sansFont = '"IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
export const monoFont = '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace';

export const colorTokens = {
  light: {
    canvas: '#f4f4f4', surface: '#ffffff', subtleSurface: '#f4f4f4',
    textPrimary: '#161616', textSecondary: '#525252', textMuted: '#6f6f6f',
    borderSubtle: '#e0e0e0', controlBorder: '#8d8d8d',
    interactive: '#0f62fe', selection: '#edf5ff', selectionText: '#0043ce', selectionBorder: '#d0e2ff',
    successText: '#198038', successGraphic: '#24a148', successSurface: '#defbe6',
    errorText: '#da1e28', errorSurface: '#fff1f1',
    warningText: '#8a3800', warningSurface: '#fff1e6',
    purpleText: '#6929c4', purpleSurface: '#f6f2ff', hover: '#e8e8e8',
  },
  dark: {
    canvas: '#161616', surface: '#262626', subtleSurface: '#393939',
    textPrimary: '#f4f4f4', textSecondary: '#c6c6c6', textMuted: '#a8a8a8',
    borderSubtle: '#525252', controlBorder: '#8d8d8d',
    interactive: '#78a9ff', selection: '#002d9c', selectionText: '#d0e2ff', selectionBorder: '#0043ce',
    successText: '#42be65', successGraphic: '#42be65', successSurface: '#044317',
    errorText: '#fa4d56', errorSurface: '#520408',
    warningText: '#ff832b', warningSurface: '#3e1a00',
    purpleText: '#be95ff', purpleSurface: '#31135e', hover: '#393939',
  },
};

export const categoryColors = {
  light: { gfx1250: '#0f62fe', gfx950: '#8a3800', Triton: '#8a3ffc', DeepSeek: '#007d79', TensileLite: '#0043ce', ST: '#525252', MT: '#9f1853' },
  dark: { gfx1250: '#78a9ff', gfx950: '#ff832b', Triton: '#be95ff', DeepSeek: '#3ddbd9', TensileLite: '#a6c8ff', ST: '#c6c6c6', MT: '#ff7eb6' },
};

export function categoryColor(mode, label, fallback) {
  const colors = categoryColors[mode];
  return Object.hasOwn(colors, label) ? colors[label] : fallback;
}
