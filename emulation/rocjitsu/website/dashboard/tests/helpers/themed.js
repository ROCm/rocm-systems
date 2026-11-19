import { createElement } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { createDashboardTheme } from '../../src/theme/theme.js';

// createElement(type, props) wrapped in the real dashboard theme, for components that read theme.palette.dashboard.
export const themed = (type, props, mode = 'light') => createElement(ThemeProvider, { theme: createDashboardTheme(mode) }, createElement(type, props));
