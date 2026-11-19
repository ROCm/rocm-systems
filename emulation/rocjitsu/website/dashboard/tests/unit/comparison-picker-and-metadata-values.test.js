import { cloneElement, createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import { TextField } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { createDashboardTheme } from '../../src/theme/theme.js';
import RunMetadataDiff from '../../src/components/compare/RunMetadataDiff.jsx';
import BranchEnvironment from '../../src/components/branch/BranchEnvironment.jsx';
import { formatFullDate } from '../../src/utils/formatters.js';
import RunSelector from '../../src/components/compare/RunSelector.jsx';

// Exercise the real MUI DOM/styles in Node, injecting the FormControl focus
// state at the renderInput seam. Browser event/box measurements belong to lead.
const state = vi.hoisted(() => ({ focused: false, autocomplete: null, input: null }));
vi.mock('@mui/material', async (importOriginal) => {
  const original = await importOriginal();
  return { ...original, Autocomplete: (props) => {
    state.autocomplete = props;
    return createElement(original.Autocomplete, { ...props, renderInput: (params) => {
      const element = props.renderInput(params);
      state.input = element.props;
      return cloneElement(element, { focused: state.focused });
    } });
  } };
});
const run = { runId: 'fictional-exact-attempt', branch: 'topic/fictional', modes: ['default'], timestamp: '2026-10-05T14:00:00Z', provenance: { rocjitsuCommitSha: 'a'.repeat(40) } };
function renderSelector(label, value, options = [run], focused = false, mode = 'light') {
  state.focused = focused;
  return renderToStaticMarkup(createElement(ThemeProvider, { theme: createDashboardTheme(mode) }, createElement(RunSelector, { label, value, options, onChange() {} })));
}
function labelGeometry(html) {
  const label = html.match(/<label\b[^>]*>/)[0];
  const className = label.match(/class="([^"]*)"/)[1].split(' ').find((name) => name.startsWith('css-'));
  const css = [...html.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)].map((match) => match[1]).join('');
  const rules = [...css.matchAll(new RegExp(`\\.${className}\\{([^}]*)\\}`, 'g'))].map((match) => match[1]);
  return { shrink: label.match(/data-shrink="([^"]*)"/)[1], transforms: rules.flatMap((rule) => [...rule.matchAll(/(?:^|;)transform:([^;]*)/g)].map((match) => match[1])) };
}

test.each([
  ['Baseline run', 'light'], ['Candidate run', 'light'],
  ['Baseline run', 'dark'], ['Candidate run', 'dark'],
])('%s keeps the same label/notch geometry empty, focused and selected in %s', (label, mode) => {
  const empty = renderSelector(label, null, [run], false, mode);
  const focused = renderSelector(label, null, [run], true, mode);
  const selected = renderSelector(label, run, [run], false, mode);
  const cleared = renderSelector(label, null, [run], false, mode);
  const geometry = labelGeometry(empty);
  expect(geometry.shrink).toBe('true');
  expect(labelGeometry(focused)).toEqual(geometry);
  expect(labelGeometry(selected)).toEqual(geometry);
  expect(labelGeometry(cleared)).toEqual(geometry);
  expect(empty).toMatch(/<label[^>]*for="[^"]+"/);
  expect(empty).toContain('role="combobox"');
  expect(empty).toContain('value=""');
  expect(focused).toContain('Mui-focused');
  expect(selected).toMatch(/<input[^>]*value="[^"]+"/);
  expect(state.autocomplete.openOnFocus).toBe(true);
  expect(state.autocomplete.disablePortal).not.toBe(true);
  expect(state.input.slotProps.input.ref).toBeTypeOf('function');
  expect(state.input.slotProps.htmlInput.onFocus).toBeTypeOf('function');
  expect(state.input.slotProps.input.endAdornment).toBeTruthy();
  // TextField must keep MUI's anchor, handlers, adornment and label association
  // when adding its shrink override; overwriting slotProps breaks the picker.
  expect(state.input.slotProps.inputLabel.htmlFor).toBe(state.input.id);
});

test('a controlled empty selection with no published options remains disabled and labelled', () => {
  const html = renderSelector('Baseline run', null, []);
  expect(labelGeometry(html).shrink).toBe('true');
  expect(html).toMatch(/<input[^>]*disabled=""/);
  expect(html).not.toContain('selected-identity');
});

test('unforced MUI empty labels animate on focus out of flow, not by adding control margin', () => {
  const render = (focused) => renderToStaticMarkup(createElement(TextField, {
    label: 'Baseline run', size: 'small', value: '', focused,
  }));
  const idle = render(false);
  const focused = render(true);
  expect(labelGeometry(idle)).toEqual({ shrink: 'false', transforms: [
    'translate(0, 20px) scale(1)', 'translate(0, 17px) scale(1)',
    'translate(14px, 16px) scale(1)', 'translate(14px, 9px) scale(1)',
  ] });
  expect(labelGeometry(focused).shrink).toBe('true');
  expect(labelGeometry(focused).transforms.at(-1)).toBe('translate(14px, -9px) scale(0.75)');
  for (const html of [idle, focused]) {
    expect(html).toContain('position:absolute;left:0;top:0');
    expect(html).toContain('margin:0;border:0;vertical-align:top');
  }
});

const filters = { targets: ['gfx1250'], suites: ['Triton'], modes: ['default'] };
const selection = { target: 'gfx1250', mode: 'default' };
const details = [
  { key: 'literal', label: 'Literal flag', value: '-old+literal' },
  { key: 'same', label: 'Same metadata', value: { version: 0, enabled: false } },
  { key: 'baseOnly', label: 'Baseline-only fact', value: false },
];
const baseline = { ...run, tests: [], configurations: [], provenance: { ...run.provenance, details } };
const candidate = { ...baseline, runId: 'fictional-candidate', provenance: { ...run.provenance, details: [
  { ...details[0], value: '+new-literal' },
  { ...details[1], value: { version: 0, enabled: false } },
  { key: 'nextOnly', label: 'Candidate-only fact', value: 0 },
] } };
function renderMetadata(Component, mode, base = baseline, next = candidate) {
  const theme = createDashboardTheme(mode);
  const html = renderToStaticMarkup(createElement(ThemeProvider, { theme }, createElement(Component, {
    baseline: base, reference: base, candidate: next, filters, selection,
  })));
  return { html, theme };
}
function metadataRow(html, id) {
  return [...html.matchAll(/<tr\b[^>]*>[\s\S]*?<\/tr>/g)].map((match) => match[0]).find((row) => row.includes(`data-testid="${id}"`));
}
function rowStyle(html, row) {
  const className = row.match(/class="([^"]*)"/)[1].split(' ').find((name) => name.startsWith('css-'));
  const css = [...html.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)].map((match) => match[1]).join('');
  return [...css.matchAll(new RegExp(`\\.${className}[^{}]*\\{[^}]*\\}`, 'g'))].map((match) => match[0]).join('');
}
test.each([
  [RunMetadataDiff, 'metadata-row-', 'light'], [RunMetadataDiff, 'metadata-row-', 'dark'],
  [BranchEnvironment, 'branch-environment-', 'light'], [BranchEnvironment, 'branch-environment-', 'dark'],
].map(([Component, prefix, mode]) => ({ name: Component.name, Component, prefix, mode })))('metadata $name marks changed rows with a light blue fill and a left bar, $mode, without repeated status text', ({ Component, prefix, mode }) => {
  const { html, theme } = renderMetadata(Component, mode);
  const changed = metadataRow(html, `${prefix}literal`);
  const unchanged = metadataRow(html, `${prefix}same`);
  expect(changed).toContain('data-different="true"');
  expect(unchanged).toContain('data-different="false"');
  const css = rowStyle(html, changed);
  // Differences: very light blue cell fill plus a bar on the first cell only.
  expect(css).toContain(`box-shadow:inset 3px 0 0 ${theme.palette.primary.main}`);
  expect(css).toContain('>.MuiTableCell-root');
  expect(rowStyle(html, unchanged)).not.toContain('box-shadow');
  expect(rowStyle(html, unchanged)).not.toContain('>.MuiTableCell-root');
  expect(changed).toContain('Literal flag');
  expect(changed.indexOf('-old+literal')).toBeLessThan(changed.indexOf('+new-literal'));
  expect(unchanged).toContain('&quot;enabled&quot;:false');
  expect(metadataRow(html, `${prefix}baseOnly`)).toContain('data-different="true"');
  expect(metadataRow(html, `${prefix}nextOnly`)).toContain('data-different="true"');
  expect(html).not.toContain('Changed');
});

test.each([RunMetadataDiff, BranchEnvironment].map((Component) => ({ name: Component.name, Component })))('metadata $name distinguishes typed values from their display text', ({ Component }) => {
  const base = { ...baseline, provenance: { ...baseline.provenance, details: [{ key: 'typed', label: 'Typed fact', value: false }] } };
  const next = { ...candidate, provenance: { ...candidate.provenance, details: [
    { key: 'typed', label: 'Typed fact', value: 'false' },
    { key: 'missingLiteral', label: 'Literal missing text', value: 'Not provided' },
  ] } };
  const { html } = renderMetadata(Component, 'light', base, next);
  const prefix = Component === RunMetadataDiff ? 'metadata-row-' : 'branch-environment-';
  expect(metadataRow(html, `${prefix}typed`)).toContain('data-different="true"');
  expect(metadataRow(html, `${prefix}missingLiteral`)).toContain('data-different="true"');
});

test.each([RunMetadataDiff, BranchEnvironment].map((Component) => ({ name: Component.name, Component })))('metadata $name never calls candidate facts different without a baseline', ({ Component }) => {
  const { html } = renderMetadata(Component, 'light', null);
  expect(html).not.toContain('data-different="true"');
  expect(html).toContain('>—</td>');
  expect(html).toContain('fictional-candidate');
});

test('branch timestamp tooltips preserve exact values when local labels match', () => {
  const base = { ...baseline, commitTimestamp: '2026-10-05T09:00:00.123Z', timestamp: '2026-10-05T10:00:00Z' };
  const next = { ...candidate, commitTimestamp: '2026-10-05T09:00:00.456Z', timestamp: '2026-10-05T10:00:30Z' };
  const { html } = renderMetadata(BranchEnvironment, 'light', base, next);
  for (const [key, field] of [['committedAt', 'commitTimestamp'], ['startedAt', 'timestamp']]) {
    expect(formatFullDate(base[field])).toBe(formatFullDate(next[field]));
    const row = metadataRow(html, `branch-source-${key}`);
    if (key === 'startedAt') { expect(row).toContain('Execution started'); expect(row).not.toContain('Execution completed'); }
    expect(row).toContain('data-different="true"');
    const cells = [...row.matchAll(/<td\b[^>]*>[\s\S]*?<\/td>/g)].map(match => match[0]);
    expect(cells).toHaveLength(2);
    for (const [index, run] of [base, next].entries()) {
      expect(cells[index]).toContain(`title="${run[field]}"`);
      expect(cells[index]).toContain(`>${formatFullDate(run[field])}</td>`);
    }
  }
});
