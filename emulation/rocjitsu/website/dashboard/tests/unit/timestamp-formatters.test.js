import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import { formatPublicationTime, formatRunTime } from '../../src/utils/formatters.js';
import * as formatters from '../../src/utils/formatters.js';
import { runSelectionLabel } from '../../src/components/shared/runOptionPresentation.js';
import DashboardHeader from '../../src/components/layout/DashboardHeader.jsx';

const timestamp = '2026-10-05T14:07:00Z';

test('compact run formatting preserves the year-less 24-hour local table label', () => {
  expect(typeof formatters.formatCompactRunTime).toBe('function');
  for (const value of [timestamp, '2026-10-05T00:07:00Z', '2026-10-05T23:07:00-04:00']) {
    const expected = new Date(value).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false,
    });
    expect(formatters.formatCompactRunTime(value)).toBe(expected);
  }
});

test('picker formatting preserves the browser-local 24-hour label', () => {
  const expected = `${new Intl.DateTimeFormat(undefined, {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    hour12: false, timeZoneName: 'short',
  }).format(new Date(timestamp))}`;
  expect(formatRunTime(timestamp)).toBe(expected);
  expect(runSelectionLabel({ timestamp })).toBe(`- · Run ${expected}`);
});

test('publication formatting preserves the year-less local header label', () => {
  const date = new Date(timestamp);
  const expected = `${new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' }).format(date)} · ${new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZoneName: 'short' }).format(date)}`;
  expect(formatPublicationTime(timestamp)).toBe(expected);
  const html = renderToStaticMarkup(createElement(DashboardHeader, { data: { generatedAt: timestamp }, mode: 'light' }));
  expect(html).toContain(expected);
  expect(html).toContain(`dateTime="${timestamp}"`);
});

test.each([null, undefined, '', 'invalid', '2026-99-99', 1, true])('unavailable timestamp %j retains caller-specific fallbacks', (value) => {
  expect(formatters.formatCompactRunTime(value)).toBe('-');
  expect(formatRunTime(value)).toBe('—');
  expect(formatPublicationTime(value)).toBeNull();
  const html = renderToStaticMarkup(createElement(DashboardHeader, { data: { generatedAt: value }, mode: 'light' }));
  expect(html).toContain('Data unavailable');
  expect(html).not.toContain('<time');
});
