import { expect, test } from 'vitest';
import { adaptPublishedDashboardData } from '../../src/data/dashboardDataAdapter.js';
import { provenanceDetails } from '../../src/data/provenance.js';
import { escapeHtml } from '../../src/utils/formatters.js';
import { changeTone, classifyDurationChange } from '../../src/utils/performance.js';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';

test('explicit zero tolerance is gray at zero and signed otherwise', () => {
  expect(classifyDurationChange(0, 0)).toBe('neutral');
  expect(changeTone(classifyDurationChange(0, 0))).toBe('neutral');
  expect(classifyDurationChange(0.1, 0)).toBe('slower');
  expect(changeTone(classifyDurationChange(0.1, 0))).toBe('error');
  expect(classifyDurationChange(-0.1, 0)).toBe('faster');
  expect(changeTone(classifyDurationChange(-0.1, 0))).toBe('success');
});

test('escapes every HTML-significant character in published text', () => {
  expect(escapeHtml('<img src=x onerror="alert(\'1\')"> & more')).toBe(
    '&lt;img src=x onerror=&quot;alert(&#39;1&#39;)&quot;&gt; &amp; more',
  );
});

test('omits optional provenance fields that a run does not publish', () => {
  const source = createMaintainedPublication();
  source.runs.forEach((run) => {
    delete run.publication.source.message;
    run.provenance.auxiliary = {};
  });
  const { data } = adaptPublishedDashboardData(source);

  expect(data.runs.every((run) => !provenanceDetails(run.provenance).some(({ key }) => key === 'auxiliary.sdk'))).toBe(true);
  expect(data.runs.every((run) => run.provenance.commitMessage == null)).toBe(true);
  expect(data.runs.every((run) => Boolean(run.provenance.rocjitsuCommitSha))).toBe(true);
});
