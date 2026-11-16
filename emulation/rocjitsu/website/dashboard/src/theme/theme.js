import { alpha, createTheme } from '@mui/material/styles';
import { colorTokens, shapeTokens } from './tokens';
import { tableDensities } from './tableStyles';
import { generateTargetColors } from './targetColors';
import { fontSizes, fontWeights, sansFont, textStyles, typographyCssVariables } from './typography';

export function createDashboardTheme(mode, targets = []) {
  const colors = colorTokens[mode === 'dark' ? 'dark' : 'light'];
  const phoneControl = { '@media (max-width:599.95px)': { minHeight: 44, minWidth: 44 } };
  const focusRing = { outline: `2px solid ${colors.interactive}`, outlineOffset: 2 };
  const warningAlert = { backgroundColor: colors.warningSurface, color: colors.textPrimary, border: `1px solid ${colors.warningText}`, '& .MuiAlertTitle-root, & .MuiButton-root': { color: colors.warningText } };
  const infoAlert = { backgroundColor: colors.selection, color: colors.selectionText, border: `1px solid ${colors.selectionBorder}` };
  const cardShadow = mode === 'dark'
    ? `0 1px 2px ${alpha(colors.shadowColor, 0.4)}`
    : `0 1px 2px ${alpha(colors.shadowColor, 0.06)}, 0 4px 14px ${alpha(colors.shadowColor, 0.05)}`;

  return createTheme({
    palette: {
      mode,
      primary: { main: colors.interactive, dark: colors.interactiveText, contrastText: colors.onAccent },
      secondary: { main: colors.secondaryAccent },
      success: { main: colors.successText, light: colors.successGraphic },
      warning: { main: colors.warningText },
      error: { main: colors.errorText },
      background: { default: colors.canvas, paper: colors.surface },
      action: { selected: colors.selection, hover: colors.hover },
      divider: colors.borderSubtle,
      text: { primary: colors.textPrimary, secondary: colors.textSecondary, muted: colors.textMuted, disabled: alpha(colors.textPrimary, 0.38) },
      selectionText: colors.selectionText,
      dashboard: colors,
      // Target -> chart color by list position: the first seven are fixed, later ones are generated (see targetColors.js).
      targetColors: Object.fromEntries(generateTargetColors(mode === 'dark' ? 'dark' : 'light', targets)),
    },
    typography: {
      fontFamily: sansFont,
      fontSize: fontSizes.body,
      fontWeightLight: fontWeights.regular,
      fontWeightRegular: fontWeights.regular,
      fontWeightMedium: fontWeights.medium,
      fontWeightBold: fontWeights.emphasized,
      body1: textStyles.body,
      body2: textStyles.body,
      caption: textStyles.caption,
      h1: { ...textStyles.pageTitle, letterSpacing: 0 },
      h2: { ...textStyles.sectionTitle, letterSpacing: 0 },
      h3: { ...textStyles.subsectionTitle, letterSpacing: 0 },
      h4: textStyles.sectionTitle,
      h5: textStyles.dialogTitle,
      h6: textStyles.dialogTitle,
      subtitle1: textStyles.sectionTitle,
      subtitle2: textStyles.body,
      button: { ...textStyles.label, textTransform: 'none' },
      overline: { ...textStyles.captionEmphasized, letterSpacing: '0.06em' },
      numeric: textStyles.numeric,
      numericEmphasized: textStyles.numericEmphasized,
      summaryValue: textStyles.summaryValue,
    },
    shape: { borderRadius: 8 },
    shadows: Array(25).fill('none'),
    components: {
      MuiCssBaseline: { styleOverrides: {
        ':root': typographyCssVariables,
        'b, strong': { fontWeight: fontWeights.emphasized },
        body: { backgroundColor: colors.canvas, WebkitFontSmoothing: 'antialiased', '--dashboard-scrollbar-thumb': alpha(colors.controlBorder, 0.55) },
        ':focus-visible': focusRing,
      } },
      MuiButtonBase: { styleOverrides: { root: {
        // ButtonBase resets browser outlines; apply the same policy after its reset.
        '&:focus-visible, &.Mui-focusVisible': focusRing,
      } } },
      MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' }, outlined: { borderColor: colors.border, borderRadius: shapeTokens.cardRadius, boxShadow: cardShadow } } },
      MuiCard: { styleOverrides: { root: { border: `1px solid ${colors.border}`, borderRadius: shapeTokens.cardRadius, boxShadow: cardShadow } } },
      MuiLink: { styleOverrides: { root: { color: colors.interactiveText, fontSize: fontSizes.body, fontWeight: fontWeights.emphasized } } },
      MuiAlert: { styleOverrides: {
        root: { gap: '14px', borderRadius: '8px', padding: '12px 16px', marginBottom: '16px', fontSize: fontSizes.body, border: `1px solid ${colors.border}` },
        standardSuccess: { backgroundColor: colors.successSurface, color: colors.successText },
        standardWarning: warningAlert,
        outlinedWarning: warningAlert,
        standardInfo: infoAlert,
        outlinedInfo: infoAlert,
        standardError: { backgroundColor: colors.errorSurface, color: colors.errorText },
        icon: { color: 'inherit' },
      } },
      MuiButton: { styleOverrides: { root: {
        borderRadius: shapeTokens.controlRadius,
        padding: '6px 12px', fontSize: fontSizes.body, lineHeight: '20px', gap: '6px',
        '& .MuiButton-startIcon, & .MuiButton-endIcon': { margin: 0 },
        '&.Mui-disabled': { opacity: 0.45, pointerEvents: 'none', color: colors.textMuted, '&.MuiButton-outlined': { borderColor: colors.controlBorder } },
        ...phoneControl,
        '&.MuiButton-containedPrimary': { backgroundColor: colors.interactive, color: colors.onAccent, borderColor: colors.interactive, '&:hover': { backgroundColor: colors.interactiveText } },
      } } },
      MuiIconButton: { styleOverrides: { root: { color: colors.textSecondary, width: 34, height: 34, border: 0, borderRadius: '50%', '&.Mui-disabled': { opacity: 0.45, pointerEvents: 'none' }, ...phoneControl } } },
      MuiAutocomplete: { styleOverrides: { popupIndicator: { border: 0 }, groupLabel: textStyles.captionEmphasized } },
      MuiMenuItem: { styleOverrides: { root: textStyles.body } },
      MuiToggleButtonGroup: { styleOverrides: { root: { borderRadius: '8px', overflow: 'hidden' } } },
      MuiToggleButton: { styleOverrides: { root: {
        padding: '4px 11px', fontSize: fontSizes.body, lineHeight: '20px',
        ...phoneControl, color: colors.textSecondary, backgroundColor: 'transparent', borderColor: colors.controlBorder,
        '&.Mui-selected': { backgroundColor: colors.selection, color: colors.selectionText, fontWeight: fontWeights.emphasized, '&:hover': { backgroundColor: colors.selection } },
        '&:hover': { backgroundColor: colors.hover },
      } } },
      MuiInputBase: { styleOverrides: { root: { color: colors.textPrimary, '&.Mui-disabled': { color: colors.textMuted, WebkitTextFillColor: colors.textMuted } }, input: { '&::placeholder': { color: colors.textMuted, opacity: 1 } } } },
      MuiOutlinedInput: { styleOverrides: {
        root: { borderRadius: shapeTokens.controlRadius, backgroundColor: colors.surface, color: colors.textPrimary, fontSize: fontSizes.body, ...phoneControl, '&.Mui-error .MuiOutlinedInput-notchedOutline': { borderColor: colors.warningText } },
        input: { padding: '9px 12px' },
        // 4px fieldset inset + 1px border + 5px legend gap = 10px label origin.
        notchedOutline: { paddingInline: '4px', borderColor: colors.controlBorder, '& legend': { fontSize: fontSizes.caption, letterSpacing: 0 }, '& legend > span': { paddingInline: '5px', fontFamily: sansFont, fontWeight: fontWeights.regular } },
      } },
      MuiSelect: { styleOverrides: { select: { padding: '5px 10px', paddingRight: '32px', fontSize: fontSizes.body }, icon: { color: colors.textSecondary } } },
      MuiInputLabel: { styleOverrides: { root: { ...textStyles.body, color: colors.textSecondary }, shrink: { ...textStyles.caption, letterSpacing: 0, maxWidth: 'calc(100% - 20px)', backgroundColor: 'transparent', paddingInline: 0, transform: 'translate(10px, -9px)' } } },
      MuiFormHelperText: { styleOverrides: { root: { ...textStyles.caption, marginTop: '4px' } } },
      MuiLinearProgress: { styleOverrides: { root: { height: 4, backgroundColor: colors.barTrack }, bar: { backgroundColor: colors.interactive } } },
      MuiSkeleton: { styleOverrides: { root: { borderRadius: '12px', backgroundColor: colors.barTrack, '&::after': { background: `linear-gradient(90deg, transparent, ${colors.subtleSurface}, transparent)` }, '@media (prefers-reduced-motion: reduce)': { animation: 'none', '&::after': { animation: 'none' } } }, wave: { '&::after': { animationDuration: '1.4s', animationTimingFunction: 'ease-in-out' } } } },
      MuiPaginationItem: { styleOverrides: { root: { color: colors.textPrimary, '&:hover': { backgroundColor: colors.hover }, '&.Mui-selected': { backgroundColor: colors.selection } }, icon: { fontSize: 18 } } },
      MuiTabs: { styleOverrides: { indicator: { borderRadius: 2 } } },
      MuiTab: { styleOverrides: { root: { minHeight: 44, paddingInline: 12 } } },
      MuiChip: { styleOverrides: { root: {
        borderRadius: 3, fontSize: fontSizes.body, fontWeight: fontWeights.emphasized, backgroundColor: colors.subtleSurface,
      }, filled: {
        '&.MuiChip-colorSuccess': { backgroundColor: colors.successSurface, color: colors.successText },
        '&.MuiChip-colorWarning': { backgroundColor: colors.warningSurface, color: colors.warningText },
        '&.MuiChip-colorError': { backgroundColor: colors.errorSurface, color: colors.errorText },
      }, outlined: {
        backgroundColor: 'transparent', color: colors.textSecondary, borderColor: colors.controlBorder,
        '&.MuiChip-colorSuccess': { backgroundColor: 'transparent', color: colors.successText, borderColor: colors.successText },
        '&.MuiChip-colorWarning': { backgroundColor: 'transparent', color: colors.warningText, borderColor: colors.warningText },
        '&.MuiChip-colorError': { backgroundColor: 'transparent', color: colors.errorText, borderColor: colors.errorText },
      }, sizeSmall: { height: 22 } } },
      MuiTableCell: { styleOverrides: {
        root: { ...tableDensities.normal, borderBottomColor: colors.border },
        head: { color: colors.textSecondary, backgroundColor: colors.tableHeader, borderBottomColor: colors.border, ...textStyles.tableHeader },
        body: textStyles.body,
      } },
      MuiTableRow: { styleOverrides: { root: { '&:not(.MuiTableRow-head):hover': { backgroundColor: colors.hover } } } },
      MuiTooltip: { defaultProps: { arrow: true }, styleOverrides: { tooltip: textStyles.tooltip } },
    },
  });
}
