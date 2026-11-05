import { execFileSync } from 'node:child_process';
import { expect, test } from 'vitest';

// A fresh process ensures Intl formatters and Date use the same timezone from startup.
test.each([
  ['UTC', '2026-01-01', '2026-01-01'],
  ['America/Los_Angeles', '2025-12-31', '2026-01-01'],
  ['Pacific/Kiritimati', '2026-01-01', '2026-01-02'],
])('calendar grouping and timestamps agree in %s', (zone, earlyDay, lateDay) => {
  const script = `
    import assert from 'node:assert/strict';
    import { localDay } from './src/utils/calendar.js';
    import { formatShortDate, formatFullDate, formatRunTime, formatPublicationTime } from './src/utils/formatters.js';
    import { periodKey } from './src/data/selectors.js';
    import { selectOverviewProjection } from './src/data/overviewProjection.js';
    const early = '2026-01-01T00:30:00Z';
    const late = '2026-01-01T23:30:00Z';
    const days = ${JSON.stringify([earlyDay, lateDay])};
    for (const [index, timestamp] of [early, late].entries()) {
      assert.equal(localDay(timestamp), days[index]);
      assert.equal(periodKey(timestamp, 'daily'), days[index]);
      assert.equal(periodKey(timestamp, 'monthly'), days[index].slice(0, -3));
      assert.equal(formatShortDate(timestamp), formatShortDate(days[index]));
      const date = new Date(timestamp);
      const expectedFull = date.toLocaleString(undefined, {
        year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZoneName: 'short',
      });
      assert.equal(formatFullDate(timestamp), expectedFull);
      const zoneName = new Intl.DateTimeFormat(undefined, { timeZoneName: 'short' }).formatToParts(date).find(p => p.type === 'timeZoneName').value;
      assert.ok(formatRunTime(timestamp).includes(zoneName));
      assert.ok(!formatPublicationTime(timestamp).includes('UTC') || process.env.TZ === 'UTC');
    }
    assert.equal(periodKey('2026-01-05T00:30:00Z'), process.env.TZ === 'America/Los_Angeles' ? '2025-12-29' : '2026-01-05');
    const runs = [early, late].map((commitTimestamp, index) => ({
      runId: String(index), branch: 'develop', commitTimestamp, timestamp: commitTimestamp,
      provenance: { rocjitsuCommitSha: String(index) }, tests: [],
    }));
    const original = JSON.stringify(runs);
    const projection = selectOverviewProjection({ runs }, { targets: [], modes: [], suites: [] });
    assert.equal(projection.anchorDay, days[1]);
    assert.equal(projection.startDay, days[0]);
    assert.deepEqual(projection.slots.map(slot => slot.dayKey), [...new Set(days)]);
    assert.equal(JSON.stringify(runs), original);
    for (const timestamp of ['2026-03-09T06:30:00Z', '2026-11-02T07:30:00Z']) {
      const week = selectOverviewProjection({ runs: [], generatedAt: timestamp }, { targets: [] }, { range: '1W' });
      assert.equal(week.dayKeys.length, 7);
      assert.equal(new Set(week.dayKeys).size, 7);
      assert.equal(week.anchorDay, localDay(timestamp));
    }
    if (process.env.TZ === 'America/Los_Angeles') {
      assert.notEqual(formatFullDate('2026-11-01T08:30:00Z'), formatFullDate('2026-11-01T09:30:00Z'));
    }
    console.log('ok');
  `;
  expect(execFileSync(process.execPath, ['--input-type=module', '-e', script], {
    cwd: new URL('../..', import.meta.url), env: { ...process.env, TZ: zone }, encoding: 'utf8',
  }).trim()).toBe('ok');
});

test.each([[2026, 2, 9], [2026, 10, 2]])('benchmark range includes exactly seven local dates across %j-%j-%j', async (year, month, day) => {
  const { benchmarkRangeData } = await import('../../src/components/benchmarks/benchmarkExplorer.js');
  const latest = new Date(year, month, day, 12);
  const cutoff = new Date(year, month, day - 6);
  const timestamps = [new Date(cutoff.getTime() - 1), cutoff, latest].map(date => date.toISOString());
  const runs = timestamps.map(commitTimestamp => ({ commitTimestamp }));
  expect(benchmarkRangeData({ runs }, '1W').runs).toEqual(runs.slice(1));
});
