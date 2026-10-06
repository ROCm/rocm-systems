import { expect, test } from 'vitest';
import { loadDashboardData } from '../../src/data/dashboardData.js';
import { provenanceDetails } from '../../src/data/provenance.js';
import { catalogSolidLineSeries, chartGapPresentation } from '../../src/utils/chartGaps.js';
import { escapeHtml } from '../../src/utils/formatters.js';
import { changeTone, classifyDurationChange } from '../../src/utils/performance.js';
import { cloneBenchmarkData } from '../fixtures/publishedData.js';

test('duration change color is gray at zero and signed otherwise', () => {
  expect(classifyDurationChange(0)).toBe('neutral');
  expect(changeTone(classifyDurationChange(0))).toBe('neutral');
  expect(classifyDurationChange(0.1)).toBe('slower');
  expect(changeTone(classifyDurationChange(0.1))).toBe('error');
  expect(classifyDurationChange(-0.1)).toBe('faster');
  expect(changeTone(classifyDurationChange(-0.1))).toBe('success');
});

test('escapes every HTML-significant character in published text', () => {
  expect(escapeHtml('<img src=x onerror="alert(\'1\')"> & more')).toBe(
    '&lt;img src=x onerror=&quot;alert(&#39;1&#39;)&quot;&gt; &amp; more',
  );
});

test('interpolates only the dotted bridge across unavailable chart values', () => {
  const presentation = chartGapPresentation([{ value: 10 }, null, null, { value: 16 }]);
  expect(presentation.estimatedValues).toEqual([10, 12, 14, 16]);
  expect(presentation.segments).toEqual([[10, 12, 14, 16]]);
});

test('does not interpolate dotted bridges across hard chart breaks', () => {
  const presentation = chartGapPresentation([{ value: 10 }, null, { value: 30 }], [1]);
  expect(presentation.estimatedValues).toEqual([10, 30, 30]);
  expect(presentation.segments).toEqual([[null, 30, 30]]);
});

test('keeps the first measured vertex of each catalog on the solid series', () => {
  const points = [
    { value: 10, run: { catalogId: 'v1' } },
    { value: 12, run: { catalogId: 'v1' } },
    { value: 20, run: { catalogId: 'v2' } },
  ];
  const segments = catalogSolidLineSeries(points, [2]);
  expect(segments).toHaveLength(2);
  expect(segments[0].showSymbol).toBe(false);
  expect(segments[0].data[1]).toMatchObject({ value: 12 });
  expect(segments[0].data[2]).toBeNull();
  expect(segments[1].showSymbol).toBe(true);
  expect(segments[1].data[2]).toMatchObject({ value: 20 });
  expect(segments[1].data[1]).toBeNull();
});

test('omits optional provenance fields that a run does not publish', () => {
  const rawData = cloneBenchmarkData();
  rawData.runs.forEach((run) => {
    delete run.provenance.commitMessage;
    delete run.provenance.details;
    delete run.provenance.rocmSdkVersion;
    delete run.provenance.pythonVersion;
    delete run.provenance.torchVersion;
    delete run.provenance.tritonCommitSha;
    delete run.provenance.tensileLiteCommitSha;
  });
  const data = loadDashboardData(rawData);

  expect(data.runs.every((run) => provenanceDetails(run.provenance).length === 0)).toBe(true);
  expect(data.runs.every((run) => !Object.hasOwn(run.provenance, 'commitMessage'))).toBe(true);
  expect(data.runs.every((run) => Boolean(run.provenance.rocjitsuCommitSha))).toBe(true);
});
