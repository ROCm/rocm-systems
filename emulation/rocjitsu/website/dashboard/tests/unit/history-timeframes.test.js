import { expect, test } from 'vitest';
import { readDashboardRoute } from '../../src/hooks/useDashboardState.js';
import { selectOverview } from '../../src/data/selectors.js';
import { benchmarkRangeData } from '../../src/components/benchmarks/benchmarkExplorer.js';
import { benchmarkData } from '../fixtures/publishedData.js';

const registryPath = '../../src/config/historyRanges.js';
const expectedRanges = [
  { id: '1W', days: 7, label: 'Trailing 7 days', period: 'past week' },
  { id: '1M', days: 30, label: 'Trailing 30 days', period: 'past month' },
  { id: '3M', days: 90, label: 'Trailing 90 days', period: 'past 3 months' },
  { id: 'ALL', days: null, label: 'All available history', period: 'all available history' },
];

test('visible history ranges keep their order, labels and commit-window lengths', async () => {
  await expect(import(registryPath)).resolves.toMatchObject({ historyRanges: expectedRanges });
});

test.each(expectedRanges)('$id is accepted unchanged in dashboard URLs', ({ id }) => {
  expect(readDashboardRoute(`https://example.test/?range=${id}`)).toMatchObject({ historyRange: id, routeError: '' });
});

test.each(['1D', '6M', 'YTD'])('%s remains selector-only rather than a URL or benchmark range', (range) => {
  const filters = { targets: benchmarkData.targets, suites: benchmarkData.suites, modes: benchmarkData.modes };
  expect(readDashboardRoute(`https://example.test/?range=${range}`)).toMatchObject({ historyRange: 'ALL', routeError: 'Invalid range selection in this URL.' });
  expect(selectOverview(benchmarkData, filters, range).history.slots.length).toBeGreaterThan(0);
  expect(benchmarkRangeData(benchmarkData, range)).toBe(benchmarkData);
});

test.each(['unknown', null, undefined])('%s resolves to all history in both consumers', (range) => {
  const filters = { targets: benchmarkData.targets, suites: benchmarkData.suites, modes: benchmarkData.modes };
  expect(selectOverview(benchmarkData, filters, range)).toEqual(selectOverview(benchmarkData, filters, 'ALL'));
  expect(benchmarkRangeData(benchmarkData, range)).toBe(benchmarkData);
});

test.each(['ALL', 'unknown', null])('%s does not slice benchmark measurements', (range) => {
  expect(benchmarkRangeData(benchmarkData, range)).toBe(benchmarkData);
});
