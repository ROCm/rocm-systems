import { expect, test } from 'vitest';
import { loadDashboardData } from '../../src/data/dashboardData.js';
import { provenanceDetails } from '../../src/data/provenance.js';
import { chartGapPresentation } from '../../src/utils/chartGaps.js';
import { escapeHtml } from '../../src/utils/formatters.js';
import { cloneBenchmarkData } from '../fixtures/publishedData.js';

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
