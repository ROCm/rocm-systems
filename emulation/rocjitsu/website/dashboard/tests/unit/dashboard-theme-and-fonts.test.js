import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { createDashboardTheme } from '../../src/theme/theme';

describe('v1 bundled fonts', () => {
  it('imports the OFL license assets into the theme for Vite production delivery', () => {
    const theme = createDashboardTheme('light');
    expect(theme.fontLicenses).toEqual({ sans: expect.stringContaining('inter-OFL.txt'), mono: expect.stringContaining('ibmplexmono-OFL.txt') });
    const source = readFileSync(new URL('../../src/theme/theme.js', import.meta.url), 'utf8');
    expect(source).toContain('inter-OFL.txt?url');
    expect(source).toContain('ibmplexmono-OFL.txt?url');
  });

  it('serves every @font-face from an existing bundled file, never a remote URL', () => {
    const stylesheetUrl = new URL('../../src/index.css', import.meta.url);
    const stylesheet = readFileSync(stylesheetUrl, 'utf8');
    const faces = [...stylesheet.matchAll(/@font-face\s*\{([^}]+)\}/g)].map((match) => match[1]);
    expect(faces.length).toBeGreaterThan(0);
    for (const face of faces) {
      expect(face).toContain('font-display: swap');
      const path = face.match(/url\(['"]?([^'"()]+)['"]?\)/)[1];
      expect(existsSync(new URL(path, stylesheetUrl)), path).toBe(true);
    }
    expect(stylesheet).not.toMatch(/https?:\/\//);
  });
});

describe('dashboard theme', () => {
  it('light and dark define the same dashboard color roles', () => {
    const light = createDashboardTheme('light').palette.dashboard;
    const dark = createDashboardTheme('dark').palette.dashboard;
    expect(Object.keys(dark).sort()).toEqual(Object.keys(light).sort());
  });

  it.each(['light', 'dark'])('%s keeps a visible focus ring above the MUI ButtonBase reset', (mode) => {
    const theme = createDashboardTheme(mode);
    expect(theme.components.MuiButtonBase?.styleOverrides?.root?.['&:focus-visible, &.Mui-focusVisible'])
      .toEqual({ outline: `2px solid ${theme.palette.primary.main}`, outlineOffset: 2 });
  });
});
