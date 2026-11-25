import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import * as performance from '../../src/utils/performance.js';
import { selectRunComparison } from '../../src/data/selectors.js';
import ComparisonMetrics from '../../src/components/compare/ComparisonMetrics.jsx';

const { classifyDurationChange } = performance;

test('the named dashboard duration-change band is three percentage points', () => {
  expect(performance.DURATION_CHANGE_BAND_PERCENT).toBe(3);
});

test.each([
  { value: -3.01, state: 'faster' },
  { value: -3, state: 'neutral' },
  { value: -0.1, state: 'neutral' },
  { value: 0, state: 'neutral' },
  { value: 0.1, state: 'neutral' },
  { value: 3, state: 'neutral' },
  { value: 3.01, state: 'slower' },
  { value: null, state: 'unavailable' },
  { value: undefined, state: 'unavailable' },
  { value: NaN, state: 'unavailable' },
  { value: Infinity, state: 'unavailable' },
  { value: -Infinity, state: 'unavailable' },
])('default band classifies $value as $state', ({ value, state }) => {
  expect(classifyDurationChange(value)).toBe(state);
});

test.each([
  { value: 0.1, tolerance: 0, state: 'slower' },
  { value: -0.1, tolerance: 0, state: 'faster' },
  { value: 0, tolerance: 0, state: 'neutral' },
  { value: 5, tolerance: 5, state: 'neutral' },
  { value: -5, tolerance: 5, state: 'neutral' },
  { value: 5.01, tolerance: 5, state: 'slower' },
  { value: -5.01, tolerance: 5, state: 'faster' },
])('explicit tolerance $tolerance classifies $value as $state', ({ value, tolerance, state }) => {
  expect(classifyDurationChange(value, tolerance)).toBe(state);
});

test('the model owns metric tolerance, with a named fallback for older models', () => {
  const model = selectRunComparison(run(104), run(100), filters, 5);
  expect(model.tolerance).toBe(5);
  expect(model.counts.neutral).toBe(1);
  const html = renderToStaticMarkup(createElement(ComparisonMetrics, { model }));
  expect(html).toContain('data-testid="comparison-metric-aggregate-change" data-tone="neutral"');
  expect(html).toContain('Within ±5% noise tolerance');
  const overridden = renderToStaticMarkup(createElement(ComparisonMetrics, { model, tolerance: 0 }));
  expect(overridden).toContain('data-testid="comparison-metric-aggregate-change" data-tone="neutral"');
  expect(overridden).toContain('Within ±5% noise tolerance');
  const olderModel = selectRunComparison(run(102), run(100), filters);
  delete olderModel.tolerance;
  const legacy = renderToStaticMarkup(createElement(ComparisonMetrics, { model: olderModel }));
  expect(legacy).toContain('Within ±3% noise tolerance');
});

const filters = { targets: ['gfx950'], modes: ['default'], suites: ['Triton'] };
const run = (durationSeconds) => ({ tests: [{ testId: 'case-a', logicalTestId: 'a', target: 'gfx950', mode: 'default', suite: 'Triton', status: 'completed', durationSeconds }] });

test.each([
  { duration: 97, tolerance: undefined, state: 'neutral' },
  { duration: 103, tolerance: undefined, state: 'neutral' },
  { duration: 96, tolerance: undefined, state: 'faster' },
  { duration: 104, tolerance: undefined, state: 'slower' },
  { duration: 101, tolerance: 0, state: 'slower' },
  { duration: 104, tolerance: 5, state: 'neutral' },
  { duration: 0, baseline: 0, tolerance: undefined, state: 'unavailable' },
])('selector and metric agree for duration $duration and tolerance $tolerance', ({ duration, baseline = 100, tolerance, state }) => {
  const model = selectRunComparison(run(duration), run(baseline), filters, tolerance);
  expect(model.counts).toEqual({ faster: 0, slower: 0, neutral: 0, unavailable: 0, [state]: 1 });
  const html = renderToStaticMarkup(createElement(ComparisonMetrics, { model }));
  const tone = state === 'faster' ? 'success' : state === 'slower' ? 'error' : 'neutral';
  expect(html).toContain(`data-testid="comparison-metric-aggregate-change" data-tone="${tone}"`);
  if (state === 'neutral') expect(html).toContain(`Within ±${tolerance ?? 3}% noise tolerance`);
});
