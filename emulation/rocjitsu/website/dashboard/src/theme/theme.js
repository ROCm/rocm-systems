import { createTheme } from '@mui/material/styles';
import { colorTokens, sansFont } from './tokens';
import sansLicenseUrl from '../assets/fonts/ibmplexsans-OFL.txt?url&no-inline';
import monoLicenseUrl from '../assets/fonts/ibmplexmono-OFL.txt?url&no-inline';

export function createDashboardTheme(mode) {
  const colors = colorTokens[mode === 'dark' ? 'dark' : 'light'];
  const phoneControl = { '@media (max-width:599.95px)': { minHeight: 44, minWidth: 44 } };

  return createTheme({
    // Vite emits the licenses with the font assets, even when publicDir is false.
    fontLicenses: { sans: sansLicenseUrl, mono: monoLicenseUrl },
    palette: {
      mode,
      primary: { main: colors.interactive },
      secondary: { main: mode === 'dark' ? '#3ddbd9' : '#007d79' },
      success: { main: colors.successText },
      warning: { main: colors.warningText },
      error: { main: colors.errorText },
      background: { default: colors.canvas, paper: colors.surface },
      action: { selected: colors.selection, hover: colors.hover },
      divider: colors.borderSubtle,
      text: { primary: colors.textPrimary, secondary: colors.textSecondary, muted: colors.textMuted },
      selectionText: colors.selectionText,
      dashboard: colors,
    },
    typography: {
      fontFamily: sansFont,
      fontSize: 14,
      body1: { fontSize: '14px', lineHeight: '20px' },
      body2: { fontSize: '14px', lineHeight: '20px' },
      caption: { fontSize: '12px', lineHeight: '18px' },
      h1: { fontSize: '24px', lineHeight: '32px', fontWeight: 500, letterSpacing: 0 },
      h2: { fontSize: '16px', lineHeight: '24px', fontWeight: 600, letterSpacing: 0 },
      h3: { fontSize: '16px', lineHeight: '24px', fontWeight: 600, letterSpacing: 0 },
      button: { fontSize: '14px', fontWeight: 500, textTransform: 'none' },
      overline: { fontSize: '11px', lineHeight: '16px', fontWeight: 500, letterSpacing: '0.06em' },
    },
    shape: { borderRadius: 8 },
    shadows: Array(25).fill('none'),
    components: {
      MuiCssBaseline: { styleOverrides: {
        body: { backgroundColor: colors.canvas, WebkitFontSmoothing: 'antialiased' },
        ':focus-visible': { outline: `2px solid ${colors.interactive}`, outlineOffset: 2 },
      } },
      MuiButtonBase: { styleOverrides: { root: {
        '&:focus-visible, &.Mui-focusVisible': { outline: `2px solid ${colors.interactive}`, outlineOffset: 2 },
      } } },
      MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } },
      MuiCard: { styleOverrides: { root: { border: `1px solid ${colors.borderSubtle}`, boxShadow: 'none' } } },
      MuiButton: { styleOverrides: { root: {
        borderRadius: 6,
        ...phoneControl,
        '&.MuiButton-containedPrimary': { backgroundColor: '#0f62fe', color: '#ffffff', '&:hover': { backgroundColor: '#0043ce' } },
      } } },
      MuiIconButton: { styleOverrides: { root: { color: colors.textSecondary, ...phoneControl } } },
      MuiToggleButton: { styleOverrides: { root: phoneControl } },
      MuiOutlinedInput: { styleOverrides: {
        root: { borderRadius: 6, backgroundColor: colors.surface, ...phoneControl },
        notchedOutline: { borderColor: colors.controlBorder },
      } },
      MuiInputLabel: { styleOverrides: { root: { color: colors.textSecondary } } },
      MuiTabs: { styleOverrides: { indicator: { borderRadius: 2 } } },
      MuiTab: { styleOverrides: { root: { minHeight: 44, paddingInline: 12 } } },
      MuiChip: { styleOverrides: { root: { borderRadius: 3, fontSize: 12, fontWeight: 500 }, sizeSmall: { height: 22 } } },
      MuiTableCell: { styleOverrides: {
        head: { color: colors.textSecondary, fontSize: 12, fontWeight: 500 },
        body: { fontSize: 13 },
      } },
      MuiTooltip: { defaultProps: { arrow: true } },
    },
  });
}
