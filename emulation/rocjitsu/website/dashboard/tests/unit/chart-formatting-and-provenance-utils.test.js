import { expect, test } from 'vitest';
import { validatePublishedDashboardData } from '../../src/data/dashboardValidation.js';
import { provenanceDetails } from '../../src/data/provenance.js';
import { escapeHtml } from '../../src/utils/formatters.js';
import { changeTone, classifyDurationChange } from '../../src/utils/performance.js';
import { createSchema2Publication } from '../fixtures/schema2Dataset.js';

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

test('omits optional provenance fields that a run does not publish', () => {
  const source = createSchema2Publication();
  source.runs.forEach((run) => {
    delete run.source.message;
    run.environment = [];
  });
  const { data } = validatePublishedDashboardData(source);

  expect(data.runs.every((run) => provenanceDetails(run.provenance).length === 0)).toBe(true);
  expect(data.runs.every((run) => !Object.hasOwn(run.provenance, 'commitMessage'))).toBe(true);
  expect(data.runs.every((run) => Boolean(run.provenance.rocjitsuCommitSha))).toBe(true);
});
