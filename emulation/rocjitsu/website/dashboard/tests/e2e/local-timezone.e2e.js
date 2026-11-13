import { expect, test } from '@playwright/test';
import { createMaintainedPublication } from '../fixtures/maintainedDataset.js';
import { downloadSource, installPublication, openDashboard } from './helpers/dashboard.js';
import { readChart } from './helpers/chart.js';

for (const timezoneId of ['UTC', 'America/Los_Angeles', 'Pacific/Kiritimati']) {
  test.describe(timezoneId, () => {
    test.use({ timezoneId });
    test('local timestamps preserve UTC chart days and source downloads', async ({ page }) => {
      const publication = createMaintainedPublication();
      publication.runs = publication.runs.slice(0, 1);
      const run = publication.runs[0];
      const timestamp = '2026-01-01T00:30:00Z';
      run.provenance.rocjitsu.rocjitsu_commit_timestamp = timestamp;
      run.execution_summary.timestamp = timestamp;
      run.execution_summary.finished_at = timestamp;
      publication.index.generatedAt = timestamp;
      publication.index.runFiles = [`runs/default-branch/${run.publication.id}.json`];
      await installPublication(page, { publication });
      await openDashboard(page, '/?targets=gfx1250&modes=single&range=ALL');
      const expected = await page.evaluate((value) => {
        const date = new Date(value);
        return {
          compact: date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }),
          full: date.toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false, timeZoneName: 'short' }),
          publication: `${date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} · ${date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZoneName: 'short' })}`,
        };
      }, timestamp);
      const times = page.getByTestId('recent-runs').locator('time');
      await expect(times).toHaveCount(2);
      for (const time of await times.all()) {
        await expect(time).toHaveText(expected.compact);
        await expect(time).toHaveAttribute('title', expected.full);
        await expect(time).toHaveAttribute('datetime', timestamp);
      }
      await expect(page.locator('header time')).toHaveText(expected.publication);
      const trend = page.getByTestId('performance-trend');
      const chart = trend.getByRole('img');
      const axisLabel = await readChart(chart, (instance) => instance.getOption().xAxis[0].axisLabel.formatter(0));
      expect(axisLabel).toBe('Jan 1');
      await trend.getByRole('group', { name: 'Inspect performance trend', exact: true }).focus();
      await page.keyboard.press('End');
      await expect(trend.getByTestId('trend-inspection-status')).toContainText('2026-01-01 UTC');
      await expect(trend.getByTestId('trend-inspection-status')).toContainText(expected.full);
      expect(await downloadSource(page)).toEqual(publication);
    });
  });
}
