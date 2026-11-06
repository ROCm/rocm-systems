import { expect } from '@playwright/test';
import { benchmarkData } from '../../fixtures/publishedData.js';
import { formatRunTime } from '../../../src/utils/formatters.js';
import { runSelectionLabel, runSelectionLabelWithBranch } from '../../../src/components/shared/runOptionPresentation.js';

const labels = { candidate: 'Candidate', reference: 'Reference', 'candidate-run': 'Candidate run', 'baseline-run': 'Baseline run' };

// Run pickers show "<short SHA> · Run <time>" in their own input, not the run ID.
export const runPicker = (page, kind) => page.getByRole('combobox', { name: labels[kind], exact: true });

// Run Comparison's pickers (candidate-run, baseline-run) also show the run's branch; the Branch page pickers do not.
const labelFor = (kind, run) => kind.endsWith('-run') ? runSelectionLabelWithBranch(run) : runSelectionLabel(run);

// `expected` is a published run ID (matched by its exact label, looked up in the shared fixture or in `runs`)
// or `{ label }` for the exact text of anything else, e.g. unavailableAttempt(id) for a missing run.
// An ID that cannot be resolved throws instead of passing on any non-empty value.
// The picker's text cannot tell two runs with the same label apart, so asserting it for one of them proves nothing.
const assertLabelIsUnique = (kind, run, runs) => {
  const label = labelFor(kind, run);
  const twin = runs.find((other) => other.runId !== run.runId && labelFor(kind, other) === label);
  if (twin) throw new Error(`expectRunPicker: ${run.runId} and ${twin.runId} share the label "${label}", so the picker text cannot identify either`);
};

// `identityProvedElsewhere`: the caller also checks the run by ID (for example in the URL), so a label shared with
// another run is acceptable here.
export async function expectRunPicker(page, kind, expected, runs = benchmarkData.allRuns, { identityProvedElsewhere = false } = {}) {
  if (typeof expected === 'object' && expected !== null) {
    await expect(runPicker(page, kind)).toHaveValue(expected.label);
    return;
  }
  const run = runs.find((item) => item.runId === expected);
  if (!run) throw new Error(`expectRunPicker: ${expected} is not among the given runs; pass { label } or the publication's runs`);
  if (!identityProvedElsewhere) assertLabelIsUnique(kind, run, runs);
  // The browser can use a different timezone from the Node test runner.
  const browserTime = await page.evaluate((timestamp) => {
    if (typeof timestamp !== 'string' || !Number.isFinite(Date.parse(timestamp))) return '—';
    return new Intl.DateTimeFormat(undefined, {
      year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
      hour12: false, timeZoneName: 'short',
    }).format(new Date(timestamp));
  }, run.timestamp);
  await expect(runPicker(page, kind)).toHaveValue(labelFor(kind, run).replace(formatRunTime(run.timestamp), browserTime));
}

export const unavailableAttempt = (runId) => ({ label: `Unavailable attempt · ${runId}` });
