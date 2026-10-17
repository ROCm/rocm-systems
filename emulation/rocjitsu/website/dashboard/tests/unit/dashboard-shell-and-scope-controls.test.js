import { fixtureRunPath } from '../fixtures/runPath.js';
import { Children, createElement, isValidElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ThemeProvider } from '@mui/material';
import { afterEach, describe, expect, it, vi } from 'vitest';
import FiltersBar from '../../src/components/layout/FiltersBar.jsx';
import DashboardShell from '../../src/components/layout/DashboardShell.jsx';
import DashboardHeader from '../../src/components/layout/DashboardHeader.jsx';
import SectionCard from '../../src/components/shared/SectionCard.jsx';
import CategoryTag from '../../src/components/shared/CategoryTag.jsx';
import { createDashboardTheme } from '../../src/theme/theme.js';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

it.each([
  ['compare', false, false],
  ['overview', false, true],
  ['benchmarks', false, true],
  ['compare', true, true],
])('UI-003 gates %s scope using its own published runs (loading=%s)', (tab, loading, disabled) => {
  const publication = createSchema2Publication();
  publication.runs = publication.runs.filter(({ source }) => source.branch !== 'develop');
  publication.index.runFiles = publication.runs.map(fixtureRunPath);
  const branchOnly = validatePublishedDashboardData(publication).data;
  expect(branchOnly.runs).toHaveLength(0);
  expect(branchOnly.allRuns).toHaveLength(20);
  const html = markup(DashboardShell, { data: branchOnly, state: { ...state, tab }, loading }, 'light', true);
  const inputs = html.match(/<input\b[^>]*type="checkbox"[^>]*>/g);
  expect(inputs).toHaveLength(6);
  expect(inputs.every((input) => /\bdisabled=""/.test(input))).toBe(disabled);
});

const data = { runs: [{}], targets: ['gfx1250', 'gfx950'], suites: ['Triton', 'DeepSeek'], modes: ['ST', 'MT'] };
const state = { targets: ['gfx1250'], suites: ['Triton', 'DeepSeek'], modes: ['ST', 'MT'], setTargets() {}, setSuites() {}, setModes() {}, tab: 'overview', setTab() {} };
function markup(Component, props, mode = 'light', desktop = false) {
  const theme = createDashboardTheme(mode);
  theme.components.MuiUseMediaQuery = { defaultProps: { ssrMatchMedia: () => ({ matches: desktop }) } };
  return renderToStaticMarkup(createElement(ThemeProvider, { theme }, createElement(Component, props)));
}
function descendants(node, predicate) {
  const found = [];
  function visit(element) {
    if (!isValidElement(element)) return;
    if (predicate(element)) found.push(element);
    Children.forEach(element.props.children, visit);
  }
  Children.forEach(node, visit);
  return found;
}
function filterControls(props, label) {
  const tree = FiltersBar(props);
  const group = descendants(tree, (element) => element.props.label === label)[0];
  if (!group) return [];
  return descendants(group.type(group.props), (element) => Boolean(element.props.control)).map((element) => element.props.control);
}

afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); });

it.each(['light', 'dark'])('sidebar uses a distinct neutral surface and 2px divider in %s theme', (mode) => {
  const theme = createDashboardTheme(mode);
  const html = markup(DashboardShell, { data, state }, mode, true);
  const sidebarClass = html.match(/<aside\b[^>]*class="([^"]+)"/)[1].split(' ').find((name) => name.startsWith('css-'));
  const styles = [...html.matchAll(/<style[^>]*>(.*?)<\/style>/g)].map((match) => match[1]);
  const sidebarStyle = styles.find((style) => style.includes(`.${sidebarClass}{`));
  expect(theme.palette.action.hover).not.toBe(theme.palette.background.default);
  expect(sidebarStyle.includes(`background-color:${theme.palette.action.hover};`)).toBe(true);
  expect(sidebarStyle.includes(`border-right:2px solid ${theme.palette.divider};`)).toBe(true);
});

describe('v10 section surfaces', () => {
  it('keeps readable 12px metadata, semantic section headings and caller content styles', () => {
    const html = markup(SectionCard, { title: 'Performance Trend', subtitle: 'Selected benchmark scope', contentSx: { mt: '7px' }, children: 'Measured values' });
    expect(html.includes('font-size:12px')).toBe(true);
    expect(html.includes('font-size:11px')).toBe(false);
    expect(html).toMatch(/<h2[^>]*>Performance Trend<\/h2>/);
    expect(html.includes('margin-top:7px')).toBe(true);
    expect(html.includes('Measured values')).toBe(true);
  });
});

describe('v10 publication header', () => {
  it('executes reload/theme callbacks and exports the unmodified raw JSON envelope', async () => {
    const onReloadData = vi.fn();
    const onToggleMode = vi.fn();
    const raw = { index: { generatedAt: '2026-10-05T12:00:00Z', runFiles: ['runs/default-branch/unit-attempt.json'] }, runs: [{ schemaVersion: 2, id: 'unit-attempt' }] };
    const anchor = { click: vi.fn() };
    const createObjectURL = vi.fn(() => 'blob:unit-export');
    const revokeObjectURL = vi.fn();
    vi.stubGlobal('document', { createElement: vi.fn(() => anchor) });
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL });
    const tree = DashboardHeader({ data, downloadData: raw, loading: false, mode: 'dark', onReloadData, onToggleMode });
    const action = (label) => descendants(tree, (element) => element.props['aria-label'] === label)[0];
    action('Reload all data').props.onClick();
    action('Use light theme').props.onClick();
    action('Download JSON').props.onClick();
    expect(onReloadData).toHaveBeenCalledOnce();
    expect(onToggleMode).toHaveBeenCalledOnce();
    expect(anchor.click).toHaveBeenCalledOnce();
    expect(anchor.href).toBe('blob:unit-export');
    expect(anchor.download).toBe('rocjitsu-simulation-benchmark-data.json');
    const blob = createObjectURL.mock.calls[0][0];
    expect(blob.type).toBe('application/json');
    expect(JSON.parse(await blob.text())).toEqual(raw);
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:unit-export');
  });

  it.each([{ loading: true }, { dataError: new Error('Unavailable snapshot') }, { data: { generatedAt: 'not-a-timestamp' } }])('shows honest unavailable/loading freshness rather than a fabricated timestamp: %j', (override) => {
    const html = markup(DashboardHeader, { data, mode: 'light', ...override });
    expect(html.includes('Data as of')).toBe(false);
    expect(html.includes('<strong')).toBe(false);
    expect(html.includes(override.loading ? 'Loading run data…' : 'Data unavailable')).toBe(true);
  });

  it('disables stale snapshot downloads while loading without disabling theme switching', () => {
    const tree = DashboardHeader({ data, downloadData: { index: 'old snapshot' }, loading: true, mode: 'light' });
    const action = (label) => descendants(tree, (element) => element.props['aria-label'] === label)[0];
    expect(action('Download JSON').props.disabled).toBe(true);
    expect(action('Reload all data').props.disabled).toBe(true);
    expect(action('Use dark theme').props.disabled).not.toBe(true);
  });

  it('links the published repository without guessing its host or tree layout', () => {
    const repository = 'https://example.test/source/repository?view=source';
    const html = markup(DashboardHeader, { data: { ...data, repository }, mode: 'light' });
    expect(html.includes(`href="${repository}"`)).toBe(true);
  });

  it('shows actual publication time in UTC with only the timestamp bold', () => {
    const generatedAt = '2026-10-05T16:02:00+02:00';
    const html = markup(DashboardHeader, { data: { ...data, generatedAt }, mode: 'light', tab: 'overview' });
    expect(html.includes('Oct 5 · 14:02 UTC')).toBe(true);
    expect(html.match(/<strong\b/g)).toHaveLength(1);
    const strong = html.match(/<strong\b[^>]*>([\s\S]*?)<\/strong>/)[1];
    expect(strong.includes('Oct 5 · 14:02 UTC')).toBe(true);
    expect(strong.includes('Data as of')).toBe(false);
    expect(html.includes(`dateTime="${generatedAt}"`)).toBe(true);
    expect(html.includes('font-size:24px')).toBe(true);
    expect(html.includes('font-size:16px')).toBe(false);
  });
});

describe('v10 responsive shell', () => {
  it('conspicuously labels fictional fixture measurements only in fixture mode', () => {
    vi.stubEnv('MODE', 'fixtures');
    const html = markup(DashboardShell, { data, state });
    expect(html.includes('Test fixtures — fictional measurements')).toBe(true);
    expect(html.includes('Not production data.')).toBe(true);
    vi.stubEnv('MODE', 'production');
    expect(markup(DashboardShell, { data, state }).includes('Test fixtures — fictional measurements')).toBe(false);
  });
  it.each(['light', 'dark'])('hides canonical controls on Branch Runs but retains navigation in %s theme', (mode) => {
    const canonical = markup(DashboardShell, { data, state }, mode, true);
    expect(canonical.includes('type="checkbox"')).toBe(true);
    const html = markup(DashboardShell, { data, state: { ...state, tab: 'branch' } }, mode, true);
    expect(html.includes('Branch Runs uses its local configuration matrix')).toBe(true);
    expect(html.includes('Canonical filters are not applied here')).toBe(true);
    expect(html.includes('type="checkbox"')).toBe(false);
    expect(html.includes('aria-controls="dashboard-filters"')).toBe(false);
    expect(html.match(/role="tab"/g)).toHaveLength(4);
  });

  it('starts mobile scope collapsed before reveal and keeps the full-page sidebar in normal layout', () => {
    const html = markup(DashboardShell, { data, state, children: createElement('main', null, 'Dashboard content') });
    expect(html).toMatch(/aria-expanded="false"[^>]*aria-controls="dashboard-filters"/);
    expect(html).toContain('Filters · 1 target / 2 suites / ST + MT');
    expect(html).not.toContain('type="checkbox"');
    expect(html).not.toContain('Latest commit run');
    expect(html).not.toContain('Check all');
    expect(html).not.toContain('position:sticky');
    expect(html).not.toMatch(/(?<!min-)height:100vh/);
    expect(html).toContain('align-self:stretch');
    expect(html).toContain('font-size:32px');
    expect(html).toContain('font-size:28px');
    expect(html).toContain('min-height:42px');
    expect(html).toContain('>Dashboard<');
    for (const icon of ['ShowChartRoundedIcon', 'AccountTreeRoundedIcon', 'GridViewRoundedIcon', 'CompareArrowsRoundedIcon']) expect(html).toContain(`data-testid="${icon}"`);
    expect(html).toContain('Dashboard content');
  });
});

describe('v10 category identity', () => {
  it.each([['target', 'gfx1250', 'square'], ['suite', 'Triton', 'circle'], ['mode', 'MT', 'triangle']])('renders a decorative %s marker in tags and filters', (kind, label, shape) => {
    const html = markup(CategoryTag, { kind, label });
    expect(html).toContain(`data-category-shape="${shape}"`);
    expect(html).toMatch(new RegExp(`aria-hidden="true"[^>]*data-category-shape="${shape}"`));
    expect(html).toContain(label);
    expect(markup(FiltersBar, { data, state })).toContain(`data-category-shape="${shape}"`);
  });
});

describe('v10 explicit scope controls', () => {
  it('uses only published availability without thread-count/name inference or Check all rows', () => {
    const onlyST = { ...data, modes: ['ST'] };
    expect(filterControls({ data: onlyST, state }, 'Execution modes')).toHaveLength(1);
    const noModes = { ...data, modes: [], runs: [{ tests: [{ name: 'MT workload' }] }] };
    expect(filterControls({ data: noModes, state }, 'Execution modes')).toHaveLength(0);
    const html = markup(FiltersBar, { data: noModes, state });
    expect(html.includes('No available execution modes')).toBe(true);
    expect(html.includes('Check all')).toBe(false);
    expect(filterControls({ data, state, disabled: true }, 'Execution modes').every((control) => control.props.disabled)).toBe(true);
  });

  it('renders ST and MT as independent checkboxes and permits deliberately clearing both', () => {
    const setModes = vi.fn();
    const props = { data, state: { ...state, setModes } };
    let controls = filterControls(props, 'Execution modes');
    expect(controls).toHaveLength(2);
    expect(controls.map((control) => control.props.checked)).toEqual([true, true]);
    controls[0].props.onChange();
    expect(setModes).toHaveBeenLastCalledWith(['MT']);
    props.state.modes = ['MT'];
    controls = filterControls(props, 'Execution modes');
    controls[1].props.onChange();
    expect(setModes).toHaveBeenLastCalledWith([]);
    props.state.modes = [];
    controls = filterControls(props, 'Execution modes');
    expect(controls.map((control) => control.props.checked)).toEqual([false, false]);
    controls[0].props.onChange();
    expect(setModes).toHaveBeenLastCalledWith(['ST']);
  });
});
