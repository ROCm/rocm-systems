import { suiteColor, suiteTextColor } from '../../theme/tokens';
import { targetColor, targetTextColor } from '../../theme/targetColors';

// Marker (decorative) and text (stronger contrast) colors of a target or suite label.
export const markerColor = (theme, label, kind) => kind === 'target' ? targetColor(theme, label) : suiteColor(theme.palette.mode, label);
export const textColor = (theme, label, kind) => kind === 'target' ? targetTextColor(theme, label) : suiteTextColor(theme.palette.mode, label);
