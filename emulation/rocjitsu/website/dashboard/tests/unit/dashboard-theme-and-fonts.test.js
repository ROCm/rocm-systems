import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { createDashboardTheme } from '../../src/theme/theme';
import { fontWeights, sansFont, textStyles } from '../../src/theme/typography';

describe('native fonts', () => {
  it('declares no @font-face and ships no font files or font licenses', () => {
    const stylesheet = readFileSync(new URL('../../src/index.css', import.meta.url), 'utf8');
    expect(stylesheet).not.toMatch(/@font-face/);
    expect(stylesheet).not.toMatch(/https?:\/\//);
    expect(existsSync(new URL('../../src/assets/fonts', import.meta.url))).toBe(false);
    expect(createDashboardTheme('light').fontLicenses).toBeUndefined();
  });

  it('uses the native system sans stack, with no bundled family', () => {
    const { fontFamily } = createDashboardTheme('light').typography;
    expect(fontFamily).toBe(sansFont);
    expect(sansFont).toContain('system-ui');
    expect(sansFont).not.toMatch(/Inter|IBM Plex/);
  });

  it('defines no monospace stack; numerals use the sans with tabular figures', () => {
    expect(textStyles.numeric).toMatchObject({ fontFamily: sansFont, fontVariantNumeric: 'tabular-nums' });
    expect(textStyles.numericEmphasized).toMatchObject({ fontFamily: sansFont, fontVariantNumeric: 'tabular-nums' });
  });

  it('has three named weight tiers in increasing order, 500, 600 and 700', () => {
    expect(Object.keys(fontWeights)).toEqual(['regular', 'medium', 'emphasized']);
    expect(Object.values(fontWeights)).toEqual([500, 600, 700]);
  });

  it('gives MUI medium and the table header the shared medium tier (600, not the MUI default 500) and MUI bold the emphasized tier (700)', () => {
    const { typography } = createDashboardTheme('light');
    expect(typography.fontWeightMedium).toBe(600);
    expect(typography.fontWeightBold).toBe(700);
    expect(textStyles.tableHeader.fontWeight).toBe(600);
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
