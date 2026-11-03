import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { createDashboardTheme } from '../../src/theme/theme';

describe('native fonts', () => {
  it('declares no @font-face and ships no font files or font licenses', () => {
    const stylesheet = readFileSync(new URL('../../src/index.css', import.meta.url), 'utf8');
    expect(stylesheet).not.toMatch(/@font-face/);
    expect(stylesheet).not.toMatch(/https?:\/\//);
    expect(existsSync(new URL('../../src/assets/fonts', import.meta.url))).toBe(false);
    expect(createDashboardTheme('light').fontLicenses).toBeUndefined();
  });

  it('uses system font stacks for both families', () => {
    const { fontFamily } = createDashboardTheme('light').typography;
    expect(fontFamily).toContain('system-ui');
    expect(fontFamily).not.toMatch(/Inter|IBM Plex/);
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
