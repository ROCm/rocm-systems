import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { createDashboardTheme } from '../../src/theme/theme';
import { categoryColors, monoFont, sansFont } from '../../src/theme/tokens';
import * as sharedStyles from '../../src/theme/styles';

describe('v10 bundled fonts', () => {
  it('imports the OFL license assets into the theme for Vite production delivery', () => {
    const theme = createDashboardTheme('light');
    expect(theme.fontLicenses).toEqual({ sans: expect.stringContaining('ibmplexsans-OFL.txt'), mono: expect.stringContaining('ibmplexmono-OFL.txt') });
    const source = readFileSync(new URL('../../src/theme/theme.js', import.meta.url), 'utf8');
    expect(source).toContain('ibmplexsans-OFL.txt?url');
    expect(source).toContain('ibmplexmono-OFL.txt?url');
  });

  it('ships six local WOFF2 faces with Plex licenses through the source stylesheet', () => {
    const stylesheetUrl = new URL('../../src/index.css', import.meta.url);
    const stylesheet = readFileSync(stylesheetUrl, 'utf8');
    const faces = [...stylesheet.matchAll(/@font-face\s*\{([^}]+)\}/g)].map((match) => match[1]);
    expect(faces).toHaveLength(6);
    for (const family of ['sans', 'mono']) {
      for (const weight of [400, 500, 600]) {
        const filename = `ibm-plex-${family}-${weight}.woff2`;
        const face = faces.find((item) => item.includes(filename));
        expect(face).toBeDefined();
        expect(face).toContain(`font-weight: ${weight}`);
        expect(face).toContain('font-display: swap');
        const path = face.match(/url\(['"]?([^'"()]+)['"]?\)/)[1];
        expect(path).toBe(`./assets/fonts/${filename}`);
        const bytes = readFileSync(new URL(path, stylesheetUrl));
        expect(bytes.subarray(0, 4).toString()).toBe('wOF2');
        expect(bytes.length).toBeGreaterThan(10000);
      }
      const license = new URL(`../../src/assets/fonts/ibmplex${family}-OFL.txt`, import.meta.url);
      expect(existsSync(license)).toBe(true);
      expect(readFileSync(license, 'utf8')).toContain('SIL OPEN FONT LICENSE');
    }
    expect(stylesheet).not.toMatch(/https?:\/\//);
  });
});

describe('v10 Carbon theme', () => {
  it('exports footer-ready neutral styles for the parent-owned three-line attribution', () => {
    expect(sharedStyles.dashboardFooterStyles).toMatchObject({ borderTop: 1, borderColor: 'divider', bgcolor: 'background.default', color: 'text.secondary', px: { xs: 2, md: 3 }, py: 2.5, fontFamily: sansFont, fontSize: 12, lineHeight: '20px' });
  });
  it.each([
    ['light', '#f4f4f4', '#ffffff', '#161616', '#525252', '#e0e0e0', '#0f62fe', '#edf5ff', '#0043ce', '#198038', '#da1e28'],
    ['dark', '#161616', '#262626', '#f4f4f4', '#c6c6c6', '#525252', '#78a9ff', '#002d9c', '#d0e2ff', '#42be65', '#fa4d56'],
  ])('%s uses the approved neutral, interaction and status palette', (mode, canvas, surface, primary, secondary, divider, accent, selected, selectedText, success, error) => {
    const theme = createDashboardTheme(mode);
    expect(theme.palette.background).toEqual({ default: canvas, paper: surface });
    expect(theme.palette.text.primary).toBe(primary);
    expect(theme.palette.text.secondary).toBe(secondary);
    expect(theme.palette.divider).toBe(divider);
    expect(theme.palette.primary.main).toBe(accent);
    expect(theme.palette.action.selected).toBe(selected);
    expect(theme.palette.selectionText).toBe(selectedText);
    expect(theme.palette.success.main).toBe(success);
    expect(theme.palette.error.main).toBe(error);
    expect(theme.shape.borderRadius).toBe(8);
  });

  it.each(['light', 'dark'])('%s keeps a visible focus ring above the MUI ButtonBase reset', (mode) => {
    const theme = createDashboardTheme(mode);
    expect(theme.components.MuiButtonBase?.styleOverrides?.root?.['&:focus-visible, &.Mui-focusVisible'])
      .toEqual({ outline: `2px solid ${theme.palette.primary.main}`, outlineOffset: 2 });
  });

  it('uses Plex and productive typography instead of legacy compact typography', () => {
    const theme = createDashboardTheme('light');
    expect(sansFont).toMatch(/^"IBM Plex Sans"/);
    expect(monoFont).toMatch(/^"IBM Plex Mono"/);
    expect(theme.typography.h1).toMatchObject({ fontSize: '24px', lineHeight: '32px', fontWeight: 500 });
    expect(theme.typography.h2).toMatchObject({ fontSize: '16px', lineHeight: '24px', fontWeight: 600 });
    expect(theme.typography.body1).toMatchObject({ fontSize: '14px', lineHeight: '20px' });
    expect(theme.typography.caption).toMatchObject({ fontSize: '12px', lineHeight: '18px' });
    expect(theme.typography.overline).toMatchObject({ fontSize: '11px', lineHeight: '16px', fontWeight: 500 });
  });

  it('shares approved category colors for targets, suites and explicit simulator modes', () => {
    expect(categoryColors.light).toMatchObject({ gfx1250: '#0f62fe', gfx950: '#8a3800', Triton: '#8a3ffc', DeepSeek: '#007d79', TensileLite: '#0043ce', ST: '#525252', MT: '#9f1853' });
    expect(categoryColors.dark).toMatchObject({ gfx1250: '#78a9ff', gfx950: '#ff832b', Triton: '#be95ff', DeepSeek: '#3ddbd9', TensileLite: '#a6c8ff', ST: '#c6c6c6', MT: '#ff7eb6' });
  });
});
