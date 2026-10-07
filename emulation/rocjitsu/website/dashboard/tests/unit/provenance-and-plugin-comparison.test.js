import { expect, test } from 'vitest';
import { provenanceDetails } from '../../src/data/provenance.js';
import { selectPluginComparison, selectPluginComparisonGroups } from '../../src/data/pluginComparison.js';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

test('generic provenance preserves distinct keys even when labels repeat, including empty scalar strings', () => {
  expect(provenanceDetails({ details: [
    { key: 'sdk-a', label: 'SDK', value: 'one' }, { key: 'sdk-b', label: 'SDK', value: 'two' }, { key: 'empty', label: 'Empty fact', value: '' },
  ] })).toEqual([
    { key: 'sdk-a', label: 'SDK', value: 'one' }, { key: 'sdk-b', label: 'SDK', value: 'two' }, { key: 'empty', label: 'Empty fact', value: '' },
  ]);
});

test('reserved plugin comparison matches exact mode identities and honors explicit mode scope', () => {
  const data = validatePublishedDashboardData(createSchema2Publication()).data;
  const group = selectPluginComparisonGroups(data)[0];
  const scoped = selectPluginComparison(group, 'gfx1250', data.suites, 'vanilla', ['MT']);
  expect(scoped.rows).toHaveLength(4);
  expect(scoped.rows.every(({ test }) => test.mode === 'MT')).toBe(true);
  expect(scoped.rows.every(({ values, test }) => values.every(({ result }) => result.testId === test.testId))).toBe(true);
  expect(selectPluginComparison(group, 'gfx1250', data.suites, 'vanilla', []).rows).toEqual([]);
});

test('reserved plugin zero baselines are matched measurements, not available percentage ratios', () => {
  const source = createSchema2Publication();
  for (const run of source.runs.filter(({ comparisonId }) => comparisonId === 'fictional-develop-23')) run.configurations[0].results[0].durationSeconds = 0;
  const data = validatePublishedDashboardData(source).data;
  const result = selectPluginComparison(selectPluginComparisonGroups(data)[0], 'gfx1250', data.suites, 'vanilla', ['ST']);
  const zero = result.rows.find(({ test }) => test.logicalTestId === 'a');
  expect(zero.values.every(({ comparable }) => comparable)).toBe(true);
  expect(zero.values.every(({ delta }) => delta === null)).toBe(true);
  expect(result.summaries.every(({ comparable, counts }) => comparable === 4 && counts.unavailable === 1)).toBe(true);
});
