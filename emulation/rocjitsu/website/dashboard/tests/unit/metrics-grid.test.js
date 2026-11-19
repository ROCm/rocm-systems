import { renderToStaticMarkup } from 'react-dom/server';
import { themed } from '../helpers/themed.js';
import { expect, test } from 'vitest';
import MetricsGrid from '../../src/components/overview/MetricsGrid.jsx';
import { selectOverviewProjection } from '../../src/data/overviewProjection.js';

function cards(projection) {
  const html = renderToStaticMarkup(themed(MetricsGrid, { projection }));
  return Object.fromEntries(html.split('data-testid="metric-card-').slice(1).map((part) => {
    const name = part.slice(0, part.indexOf('"'));
    // Each row's value is its last text element (label first, value last).
    const rows = part.split('data-configuration="').slice(1).map((segment) => {
      const key = segment.slice(0, segment.indexOf('"'));
      const texts = [...segment.matchAll(/<(?:p|span)\b[^>]*>([^<]*)<\/(?:p|span)>/g)].map((match) => match[1]);
      return [key, texts.at(-1)];
    });
    return [name, { rows: Object.fromEntries(rows), text: part }];
  }));
}

const row = (key, speeds) => {
  const [target, mode] = key.split('/');
  return { key, target, mode, cohortIds: ['a', 'b'], comparedWorkloads: 2,
    latest: { completed: 1, required: 2, durationSeconds: null, speedup: null, reason: 'failed' },
    points: speeds.map((speedup, i) => ({ speedup, runId: `attempt-${i}`, commit: 'same-commit' })) };
};

test('valid data points counts finite plotted attempts per lane, including reference and excluding gaps', () => {
  const result = cards({ endpointLabel: 'Latest selected run', rows: [
    row('gfx1250/single', [1, 1.25, null, NaN, Infinity, -Infinity, undefined, '1']),
    row('gfx950/single', [1, 2, 3]),
    row('gfx1250/default', [null]),
  ] });
  expect(result['valid-data-points']?.rows).toEqual({ 'gfx1250/single': '2', 'gfx950/single': '3', 'gfx1250/default': '0' });
  expect(result['latest-results'].rows).toEqual({ 'gfx1250/single': '1/2', 'gfx950/single': '1/2', 'gfx1250/default': '1/2' });
  expect(result['latest-duration'].rows['gfx1250/single']).toBe('Unavailable');
  expect(result['relative-speedup'].rows['gfx1250/single']).toBe('Unavailable');
});

test('empty selection and empty cohorts retain their explicit states in every card', () => {
  for (const card of Object.values(cards({ rows: [], endpointLabel: 'Latest selected run' }))) {
    expect(card.text).toContain('No configurations selected');
    expect(card.rows).toEqual({});
  }
  const empty = { ...row('gfx950/single', []), cohortIds: [], latest: { completed: 0, required: 0 } };
  for (const card of Object.values(cards({ rows: [empty], endpointLabel: 'Latest selected run' }))) {
    expect(card.rows).toEqual({ 'gfx950/single': 'No shared workloads' });
    expect(card.text).not.toContain('0/0');
  }
});

const filters = { targets: ['gfx950', 'gfx1250'], modes: ['single'], suites: ['gemm', 'fft'] };
const configurations = { 'gfx950/single': ['a', 'b'], 'gfx1250/single': ['a', 'b'] };
const catalogs = Object.fromEntries(['old', 'new'].map((id) => [id, { id, configurations,
  tests: [{ id: 'a', suite: 'gemm' }, { id: 'b', suite: 'fft' }] }]));
function run(runId, day, catalogId, failed = false, completion = '12:00:00') {
  return { runId, branch: 'develop', catalogId,
    commitTimestamp: `${day}T10:00:00Z`, timestamp: `${day}T${completion}Z`,
    provenance: { rocjitsuCommitSha: day },
    configurations: filters.targets.map((target) => ({ target, thread_policy: 'single' })),
    tests: filters.targets.flatMap((target) => ['a', 'b'].map((id) => ({
      logicalTestId: id, target, mode: 'single', suite: id === 'a' ? 'gemm' : 'fft',
      status: failed && target === 'gfx1250' && id === 'b' ? 'failed' : 'completed',
      durationSeconds: failed && target === 'gfx1250' && id === 'b' ? null : 2,
    }))) };
}
const source = { catalogs, runs: [run('old', '2026-01-01', 'old'),
  run('first', '2026-01-10', 'new'), run('retry', '2026-01-10', 'new', true, '13:00:00')] };
const renderSelection = (selected = filters, options = { range: 'ALL' }) => cards(selectOverviewProjection(source, selected, options));

test('cards retain latest endpoint duration, speedup and completion without success fallback', () => {
  const result = renderSelection();
  expect(result['latest-duration'].rows).toEqual({ 'gfx1250/single': 'Unavailable', 'gfx950/single': '4.00s' });
  expect(result['relative-speedup'].rows).toEqual({ 'gfx1250/single': 'Unavailable', 'gfx950/single': '1.00×' });
  expect(result['latest-results'].rows).toEqual({ 'gfx1250/single': '1/2', 'gfx950/single': '2/2' });
  expect(result['valid-data-points'].rows).toEqual({ 'gfx1250/single': '1', 'gfx950/single': '2' });
  expect(Object.keys(result['latest-results'].rows)).toEqual(['gfx1250/single', 'gfx950/single']);
});

test('range and target changes preserve weekly all-selected-lane eligibility and separate attempts', () => {
  const weekly = renderSelection(filters, { range: '1W' });
  expect(weekly['valid-data-points'].rows).toEqual({ 'gfx1250/single': '1', 'gfx950/single': '1' });
  expect(weekly['latest-results'].rows).toEqual({ 'gfx1250/single': '2/2', 'gfx950/single': '2/2' });
  const single = renderSelection({ ...filters, targets: ['gfx950'] }, { range: '1W' });
  expect(single['valid-data-points'].rows).toEqual({ 'gfx950/single': '2' });
  expect(renderSelection(filters, { range: '1M' })['valid-data-points'].rows)
    .toEqual({ 'gfx1250/single': '1', 'gfx950/single': '2' });
});

test('suite and catalog selections recompute visible counts and completion ratios', () => {
  const gemm = renderSelection({ ...filters, suites: ['gemm'] }, { range: '1W' });
  expect(gemm['valid-data-points'].rows).toEqual({ 'gfx1250/single': '2', 'gfx950/single': '2' });
  expect(gemm['latest-results'].rows).toEqual({ 'gfx1250/single': '1/1', 'gfx950/single': '1/1' });
  const current = renderSelection(filters, { range: 'ALL', catalogId: 'new' });
  expect(current['valid-data-points'].rows).toEqual({ 'gfx1250/single': '0', 'gfx950/single': '1' });
  expect(current['latest-results'].rows).toEqual({ 'gfx1250/single': '1/2', 'gfx950/single': '2/2' });
  const empty = renderSelection({ ...filters, suites: [] });
  expect(empty['valid-data-points'].rows).toEqual({ 'gfx1250/single': 'No shared workloads', 'gfx950/single': 'No shared workloads' });
});
