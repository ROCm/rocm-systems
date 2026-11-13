import { describe, expect, test } from 'vitest';
import { adaptPublishedDashboardData, firstSeenTargets } from '../../src/data/dashboardDataAdapter.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { FIXED_TARGET_COUNT, generateTargetColors, targetColor, targetHues } from '../../src/theme/targetColors.js';
import { createDashboardTheme } from '../../src/theme/theme.js';
import { colorTokens } from '../../src/theme/tokens.js';

function luminance(hex) {
  const channels = [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16) / 255);
  return channels.map((channel) => channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4)
    .reduce((total, channel, index) => total + channel * [0.2126, 0.7152, 0.0722][index], 0);
}
const contrast = (a, b) => (Math.max(luminance(a), luminance(b)) + 0.05) / (Math.min(luminance(a), luminance(b)) + 0.05);
const hueOf = (hex) => {
  const [r, g, b] = [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16) / 255);
  const max = Math.max(r, g, b); const delta = max - Math.min(r, g, b);
  if (!delta) return 0;
  const sector = max === r ? ((g - b) / delta + 6) % 6 : max === g ? (b - r) / delta + 2 : (r - g) / delta + 4;
  return sector * 60;
};
const gap = (a, b) => Math.min(Math.abs(a - b), 360 - Math.abs(a - b));
const named = (count) => Array.from({ length: count }, (_, index) => `target-${String(index).padStart(2, '0')}`);

for (const mode of ['light', 'dark']) {
  const surface = colorTokens[mode].surface;

  describe(`${mode} generated target colors`, () => {
    test('the first two targets keep the original blue and orange', () => {
      const colors = generateTargetColors(mode, ['gfx1250', 'gfx950']);
      expect(colors.get('gfx1250')).toBe(colorTokens[mode].interactive);
      expect(colors.get('gfx950')).toBe(colorTokens[mode].seriesSecondary);
    });

    test.each([2, 3, 6, 12, 24])('%i targets get distinct colors', (count) => {
      const colors = [...generateTargetColors(mode, named(count)).values()];
      expect(new Set(colors).size).toBe(count);
    });

    test.each([3, 6, 12, 24])('every generated color of %i targets is readable on the surface (4.5:1)', (count) => {
      for (const [target, color] of generateTargetColors(mode, named(count))) {
        expect(color, target).toMatch(/^#[0-9a-f]{6}$/i);
        expect(contrast(color, surface), `${target}: ${color}`).toBeGreaterThanOrEqual(4.5);
      }
    });

    test('generated hues stay apart from each other and from the seeded blue and orange', () => {
      const hues = [...generateTargetColors(mode, named(8)).values()].map(hueOf);
      for (let a = 0; a < hues.length; a += 1) for (let b = a + 1; b < hues.length; b += 1) {
        expect(gap(hues[a], hues[b]), `${a} vs ${b}`).toBeGreaterThanOrEqual(15);
      }
    });

    test('colors follow the given first-seen order, not the names', () => {
      const colors = generateTargetColors(mode, ['gfx950', 'gfx1250']);
      expect(colors.get('gfx950')).toBe(colorTokens[mode].interactive);
      expect(colors.get('gfx1250')).toBe(colorTokens[mode].seriesSecondary);
    });

    test('the theme exposes the generated colors', () => {
      const theme = createDashboardTheme(mode, ['gfx950', 'gfx1250', 'gfx942']);
      expect(Object.keys(theme.palette.targetColors).sort()).toEqual(['gfx1250', 'gfx942', 'gfx950']);
      expect(new Set(Object.values(theme.palette.targetColors)).size).toBe(3);
    });
  });
}

test('targetHues seeds blue and orange and never loops forever on a large list', () => {
  expect(targetHues(2)).toEqual([219, 25]);
  // The seeds must stay the hues of the two palette colors the first two targets actually get.
  const [blue, orange] = targetHues(2);
  for (const mode of ['light', 'dark']) {
    expect(gap(hueOf(colorTokens[mode].interactive), blue), `${mode} blue`).toBeLessThan(8);
    expect(gap(hueOf(colorTokens[mode].seriesSecondary), orange), `${mode} orange`).toBeLessThan(8);
  }
  expect(targetHues(60)).toHaveLength(60);
});

// Inherited object keys must never resolve to Object.prototype members: they are ordinary published targets.
test.each(['constructor', 'toString', '__proto__', 'hasOwnProperty'])('literal target %s gets a usable, distinct color', (target) => {
  for (const mode of ['light', 'dark']) {
    const colors = generateTargetColors(mode, [target, 'gfx950', 'gfx1250']);
    expect(colors.get(target)).toMatch(/^#[0-9a-f]{6}$/i);
    expect(new Set(colors.values()).size).toBe(3);
    const theme = createDashboardTheme(mode, [target, 'gfx950', 'gfx1250']);
    expect(Object.hasOwn(theme.palette.targetColors, target)).toBe(true);
  }
});

test('a target the theme was not given falls back to neutral text, never another target\'s color', () => {
  const theme = createDashboardTheme('light', ['gfx1250', 'gfx950']);
  expect(targetColor(theme, 'not-published')).toBe(theme.palette.text.secondary);
  expect(Object.values(theme.palette.targetColors)).not.toContain(theme.palette.text.secondary);
});

test.each(['constructor', 'toString', 'hasOwnProperty', '__proto__'])('inherited object member %s is not a target and falls back to neutral text', (name) => {
  const theme = createDashboardTheme('light', ['gfx1250', 'gfx950']);
  expect(targetColor(theme, name)).toBe(theme.palette.text.secondary);
});

// The palette for the first seven targets is fixed. These literals are the contract: they are written out here on
// purpose, so changing the generator or the palette file cannot quietly recolor charts users already know.
const FIRST_SEVEN = {
  light: ['#0f62fe', '#c2570c', '#158463', '#ca21ca', '#5d7b14', '#dd2c42', '#158428'],
  dark: ['#78a9ff', '#ffa04d', '#50e2b8', '#e250e2', '#b8e250', '#e4586a', '#50e268'],
};

for (const mode of ['light', 'dark']) {
  describe(`${mode} fixed colors for the first seven targets`, () => {
    test('seven targets get exactly the fixed list, in order', () => {
      expect([...generateTargetColors(mode, named(7)).values()]).toEqual(FIRST_SEVEN[mode]);
      expect(FIXED_TARGET_COUNT).toBe(7);
    });

    test('the first two are still the original blue and orange tokens', () => {
      const [first, second] = FIRST_SEVEN[mode];
      expect(first).toBe(colorTokens[mode].interactive);
      expect(second).toBe(colorTokens[mode].seriesSecondary);
    });

    test('appending targets never changes a color already given, from 1 up to 24 targets', () => {
      const all = [...generateTargetColors(mode, named(24)).values()];
      for (let count = 1; count <= 24; count += 1) {
        expect([...generateTargetColors(mode, named(count)).values()], `${count} targets`).toEqual(all.slice(0, count));
      }
    });

    test('targets appearing later never recolor earlier ones, whatever they sort like', () => {
      const first = generateTargetColors(mode, ['gfx1250', 'gfx950']);
      const second = generateTargetColors(mode, ['gfx1250', 'gfx950', 'gfx1201', 'gfx1100']);
      const third = generateTargetColors(mode, ['gfx1250', 'gfx950', 'gfx1201', 'gfx1100', 'gfx942']);
      for (const [target, color] of first) expect(second.get(target)).toBe(color);
      for (const [target, color] of second) expect(third.get(target)).toBe(color);
      expect(third.get('gfx942')).toBe(FIRST_SEVEN[mode][4]);
    });

    test('duplicate published targets count once and do not shift a color', () => {
      const once = generateTargetColors(mode, ['gfx1250', 'gfx950', 'gfx942']);
      const repeated = generateTargetColors(mode, ['gfx1250', 'gfx950', 'gfx1250', 'gfx942', 'gfx950']);
      expect([...repeated]).toEqual([...once]);
    });

    test('the seven are distinct and each readable on the surface (4.5:1)', () => {
      const surface = colorTokens[mode].surface;
      expect(new Set(FIRST_SEVEN[mode]).size).toBe(7);
      for (const color of FIRST_SEVEN[mode]) expect(contrast(color, surface), color).toBeGreaterThanOrEqual(4.5);
    });

    test('targets past the seventh are generated, distinct from the fixed seven, and leave the seven alone', () => {
      const colors = [...generateTargetColors(mode, named(12)).values()];
      expect(colors.slice(0, 7)).toEqual(FIRST_SEVEN[mode]);
      expect(new Set(colors).size).toBe(12);
    });
  });
}

test('firstSeenTargets orders by first appearance, alphabetical only within one run, and never moves a target', () => {
  const run = (...targets) => ({ targets });
  const history = [run('gfx950', 'gfx1250'), run('gfx1250', 'gfx950'), run('gfx1201', 'gfx1100', 'gfx950'), run('gfx942', 'gfx1100')];
  expect(firstSeenTargets(history)).toEqual(['gfx1250', 'gfx950', 'gfx1100', 'gfx1201', 'gfx942']);
  for (let count = 1; count <= history.length; count += 1) {
    const earlier = firstSeenTargets(history.slice(0, count - 1));
    expect(firstSeenTargets(history.slice(0, count)).slice(0, earlier.length)).toEqual(earlier);
  }
});

test('the adapter exposes colors in first-seen order while the filter list stays alphabetical', () => {
  const publication = createMaintainedPublication();
  const base = publication.runs.filter((run) => run.publication.source.branch === 'develop');
  const latest = base.at(-1);
  latest.benchmark_results.push(...latest.benchmark_results.filter(({ target }) => target === 'gfx950').map((row) => ({ ...structuredClone(row), target: 'gfx942' })));
  const { data } = adaptPublishedDashboardData(publication);
  expect(data.targets).toEqual(['gfx1250', 'gfx942', 'gfx950']);
  expect(data.targetColorOrder).toEqual(['gfx1250', 'gfx950', 'gfx942']);
});

test('first appearance is decided by run time, not by the order the runs are passed in', () => {
  const run = (timestamp, ...targets) => ({ timestamp, runId: timestamp, targets });
  const history = [run('2026-10-02T00:00:00Z', 'gfx950'), run('2026-10-03T00:00:00Z', 'gfx942'), run('2026-10-01T00:00:00Z', 'gfx1250')];
  expect(firstSeenTargets(history)).toEqual(['gfx1250', 'gfx950', 'gfx942']);
  expect(firstSeenTargets([...history].reverse())).toEqual(['gfx1250', 'gfx950', 'gfx942']);
});

test('a run without a usable timestamp is ordered first, so its targets take the first colors', () => {
  const run = (timestamp, ...targets) => ({ timestamp, runId: String(timestamp), targets });
  expect(firstSeenTargets([run('2026-10-01T00:00:00Z', 'gfx1250'), run(undefined, 'gfx900')])).toEqual(['gfx900', 'gfx1250']);
});

test('side-branch runs count toward first appearance', () => {
  const publication = createMaintainedPublication();
  const branch = publication.runs.find((run) => run.publication.source.branch !== 'develop');
  const develop = publication.runs.filter((run) => run.publication.source.branch === 'develop');
  const earliest = develop.map((run) => run.execution_summary.timestamp).sort()[0];
  branch.execution_summary.timestamp = new Date(Date.parse(earliest) - 3_600_000).toISOString();
  branch.benchmark_results.forEach((row) => { row.target = 'gfx900'; });
  const { data } = adaptPublishedDashboardData(publication);
  expect(data.targetColorOrder[0]).toBe('gfx900');
});
