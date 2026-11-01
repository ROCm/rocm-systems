import { expect } from '@playwright/test';
import { benchmarkData } from '../../fixtures/publishedData.js';
import { runSelectionLabel } from '../../../src/components/shared/runOptionPresentation.js';

const labels = { candidate: 'Candidate', reference: 'Reference', 'candidate-run': 'Candidate run', 'baseline-run': 'Baseline run' };

// Run pickers show "<short SHA> · Run <time>" in their own input, not the run ID.
export const runPicker = (page, kind) => page.getByRole('combobox', { name: labels[kind], exact: true });

// Runs from the shared fixture are matched by their exact label. Runs that only exist in a test's own
// publication (or are deliberately missing) can only be checked for a non-empty selection.
export async function expectRunPicker(page, kind, runId) {
  const run = benchmarkData.allRuns.find((item) => item.runId === runId);
  if (run) await expect(runPicker(page, kind)).toHaveValue(runSelectionLabel(run));
  else await expect(runPicker(page, kind)).not.toHaveValue('');
}
