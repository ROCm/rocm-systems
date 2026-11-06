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
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { formatPublicationTime } from '../../src/utils/formatters.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';

it.each([
  ['compare', false, false],
  ['overview', false, true],
  ['benchmarks', false, true],
  ['compare', true, true],
])('UI-003 gates %s scope using its own published runs (loading=%s)', (tab, loading, disabled) => {
  const publication = createMaintainedPublication();
  publication.runs = publication.runs.filter(({ publication: { source } }) => source.branch !== 'develop');
  publication.index.runFiles = publication.runs.map(fixtureRunPath);
  const branchOnly = adaptPublishedDashboardData(publication).data;
  expect(branchOnly.runs).toHaveLength(0);
  expect(branchOnly.allRuns).toHaveLength(20);
  const html = markup(DashboardShell, { data: branchOnly, state: { ...state, tab }, loading }, 'light', true);
  const inputs = html.match(/<input\b[^>]*type="checkbox"[^>]*>/g);
  expect(inputs).toHaveLength(6);
  expect(inputs.every((input) => /\bdisabled=""/.test(input))).toBe(disabled);
});

const data = { runs: [{}], targets: ['gfx1250', 'gfx950'], suites: ['Triton', 'DeepSeek'], modes: ['single', 'default'] };
const state = { targets: ['gfx1250'], suites: ['Triton', 'DeepSeek'], modes: ['single', 'default'], setTargets() {}, setSuites() {}, setModes() {}, tab: 'overview', setTab() {} };
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

it('mobile filter summary names an empty thread-policy selection', () => {
  const html = markup(DashboardShell, { data, state: { ...state, modes: [] } });
  expect(html).toContain('no thread policies');
  expect(html).not.toContain('no modes');
});

describe('v1 section surfaces', () => {
  it('keeps readable 12px metadata, semantic section headings and caller content styles', () => {
    const html = markup(SectionCard, { title: 'Performance Trend', subtitle: 'Selected benchmark scope', contentSx: { mt: '7px' }, children: 'Measured values' });
    expect(html).toMatch(/<h2[^>]*>Performance Trend<\/h2>/);
    expect(html.includes('margin-top:7px')).toBe(true);
    expect(html.includes('Measured values')).toBe(true);
  });
});

describe('v1 publication header', () => {
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

  it('derives the source link and label from the configured repository and canonical branch', () => {
    const html = markup(DashboardHeader, { data: { ...data, repository: 'https://example.test/org/other-repo/', canonicalBranch: 'main' }, mode: 'light' });
    expect(html).toContain('href="https://example.test/org/other-repo/tree/main/emulation/rocjitsu"');
    expect(html).toContain('>org/other-repo<');
    expect(html).not.toContain('ROCm/rocm-systems');
  });

  it('omits the source link when the repository is missing or unsafe', () => {
    for (const repository of [undefined, 'javascript:alert(1)']) {
      expect(markup(DashboardHeader, { data: { ...data, repository }, mode: 'light' })).not.toContain('/tree/');
    }
  });

  it('shows local publication time with only the timestamp bold', () => {
    const generatedAt = '2026-10-05T16:02:00+02:00';
    const formatted = formatPublicationTime(generatedAt);
    const html = markup(DashboardHeader, { data: { ...data, generatedAt }, mode: 'light', tab: 'overview' });
    expect(html.includes(formatted)).toBe(true);
    expect(html.match(/<strong\b/g)).toHaveLength(1);
    const strong = html.match(/<strong\b[^>]*>([\s\S]*?)<\/strong>/)[1];
    expect(strong.includes(formatted)).toBe(true);
    expect(strong.includes('Data as of')).toBe(false);
    expect(html.includes(`dateTime="${generatedAt}"`)).toBe(true);
    expect(html.includes('font-size:24px')).toBe(true);
    expect(html.includes('font-size:16px')).toBe(false);
  });
});

describe('v1 responsive shell', () => {
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
    expect(html.includes('Branch runs uses its local configuration matrix')).toBe(true);
    expect(html.includes('Canonical filters are not applied here')).toBe(true);
    expect(html.includes('type="checkbox"')).toBe(false);
    expect(html.includes('aria-controls="dashboard-filters"')).toBe(false);
    expect(html.match(/role="tab"/g)).toHaveLength(4);
  });

  it('starts mobile scope collapsed before reveal and keeps the full-page sidebar in normal layout', () => {
    const html = markup(DashboardShell, { data, state, children: createElement('main', null, 'Dashboard content') });
    expect(html).toMatch(/aria-expanded="false"[^>]*aria-controls="dashboard-filters"/);
    expect(html).toContain('Filters · 1 target / 2 suites / single + default');
    expect(html).not.toContain('type="checkbox"');
    expect(html).not.toContain('Latest commit run');
    expect(html).not.toContain('Check all');
    expect(html).not.toMatch(/(?<!min-)height:100vh/);
    expect(html).toContain('align-self:stretch');
    for (const icon of ['ShowChartRoundedIcon', 'AccountTreeRoundedIcon', 'GridViewRoundedIcon', 'CompareArrowsRoundedIcon']) expect(html).toContain(`data-testid="${icon}"`);
    expect(html).toContain('Dashboard content');
  });
});

describe('v1 explicit scope controls', () => {
  it('uses only published availability without thread-count/name inference or Check all rows', () => {
    const onlyST = { ...data, modes: ['single'] };
    expect(filterControls({ data: onlyST, state }, 'Thread policy')).toHaveLength(1);
    const noModes = { ...data, modes: [], runs: [{ tests: [{ name: 'default workload' }] }] };
    expect(filterControls({ data: noModes, state }, 'Thread policy')).toHaveLength(0);
    const html = markup(FiltersBar, { data: noModes, state });
    expect(html.includes('No available thread policy')).toBe(true);
    expect(html.includes('Check all')).toBe(false);
    expect(filterControls({ data, state, disabled: true }, 'Thread policy').every((control) => control.props.disabled)).toBe(true);
  });

  it('renders single and default as independent checkboxes and permits deliberately clearing both', () => {
    const setModes = vi.fn();
    const props = { data, state: { ...state, setModes } };
    let controls = filterControls(props, 'Thread policy');
    expect(controls).toHaveLength(2);
    expect(controls.map((control) => control.props.checked)).toEqual([true, true]);
    controls[0].props.onChange();
    expect(setModes).toHaveBeenLastCalledWith(['default']);
    props.state.modes = ['default'];
    controls = filterControls(props, 'Thread policy');
    controls[1].props.onChange();
    expect(setModes).toHaveBeenLastCalledWith([]);
    props.state.modes = [];
    controls = filterControls(props, 'Thread policy');
    expect(controls.map((control) => control.props.checked)).toEqual([false, false]);
    controls[0].props.onChange();
    expect(setModes).toHaveBeenLastCalledWith(['single']);
  });
});
