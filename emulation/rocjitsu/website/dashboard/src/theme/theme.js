import { alpha, createTheme } from '@mui/material/styles';

export function createDashboardTheme(mode) {
  const dark = mode === 'dark';
  const background = dark ? '#0E1320' : '#F5F7FB';
  const paper = dark ? '#151C2B' : '#FFFFFF';
  const divider = dark ? '#273248' : '#E5EAF1';

  return createTheme({
    palette: {
      mode,
      primary: { main: dark ? '#818CF8' : '#5566E9' },
      secondary: { main: '#12A594' },
      success: { main: dark ? '#49C98C' : '#168A5B' },
      warning: { main: dark ? '#F5B85B' : '#B96814' },
      error: { main: dark ? '#F37B88' : '#CF4555' },
      background: { default: background, paper },
      divider,
      text: {
        primary: dark ? '#F3F6FC' : '#172033',
        secondary: dark ? '#9BA8BE' : '#647087',
      },
    },
    typography: {
      fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      h1: { fontSize: '1.9rem', fontWeight: 760, letterSpacing: '-0.035em' },
      h2: { fontSize: '1.05rem', fontWeight: 720, letterSpacing: '-0.015em' },
      h3: { fontSize: '0.95rem', fontWeight: 700 },
      button: { fontWeight: 650, textTransform: 'none' },
      overline: { fontSize: '0.67rem', fontWeight: 750, letterSpacing: '0.08em', lineHeight: 1.5 },
    },
    shape: { borderRadius: 12 },
    shadows: Array(25).fill('none').map((value, index) => index === 1
      ? dark ? '0 14px 40px rgba(0,0,0,.22)' : '0 10px 35px rgba(35,48,76,.07)'
      : value),
    components: {
      MuiCssBaseline: {
        styleOverrides: { body: { backgroundColor: background } },
      },
      MuiPaper: {
        styleOverrides: { root: { backgroundImage: 'none' } },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            border: `1px solid ${divider}`,
            boxShadow: dark ? '0 12px 32px rgba(0,0,0,.14)' : '0 10px 32px rgba(35,48,76,.055)',
          },
        },
      },
      MuiButton: {
        styleOverrides: { root: { borderRadius: 9 } },
      },
      MuiOutlinedInput: {
        styleOverrides: {
          root: {
            borderRadius: 9,
            backgroundColor: dark ? alpha('#FFFFFF', 0.025) : '#FBFCFE',
          },
        },
      },
      MuiTabs: {
        styleOverrides: { indicator: { height: 3, borderRadius: '3px 3px 0 0' } },
      },
      MuiTab: {
        styleOverrides: { root: { minHeight: 48, paddingInline: 18 } },
      },
      MuiTooltip: { defaultProps: { arrow: true } },
    },
  });
}
