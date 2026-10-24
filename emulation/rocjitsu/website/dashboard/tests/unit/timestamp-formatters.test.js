import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import { formatPublicationTime, formatRunTime } from '../../src/utils/formatters.js';
import { runSelectionLabel } from '../../src/components/shared/runOptionPresentation.js';
import DashboardHeader from '../../src/components/layout/DashboardHeader.jsx';

const timestamp = '2026-10-05T14:07:00Z';

test('picker formatting preserves the existing localized 24-hour UTC label', () => {
  const expected = `${new Intl.DateTimeFormat(undefined, {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    hour12: false, timeZone: 'UTC',
  }).format(new Date(timestamp))} UTC`;
  expect(formatRunTime(timestamp)).toBe(expected);
  expect(runSelectionLabel({ timestamp })).toBe(`- · Run ${expected}`);
});

test('publication formatting preserves the year-less UTC header label', () => {
  expect(formatPublicationTime(timestamp)).toBe('Oct 5 · 14:07 UTC');
  expect(formatPublicationTime('2026-10-05T00:07:00Z')).toBe('Oct 5 · 00:07 UTC');
  const html = renderToStaticMarkup(createElement(DashboardHeader, { data: { generatedAt: timestamp }, mode: 'light' }));
  expect(html).toContain('Oct 5 · 14:07 UTC');
  expect(html).toContain(`dateTime="${timestamp}"`);
});

test.each([null, undefined, '', 'invalid', '2026-99-99', 1, true])('unavailable timestamp %j retains caller-specific fallbacks', (value) => {
  expect(formatRunTime(value)).toBe('—');
  expect(formatPublicationTime(value)).toBeNull();
  const html = renderToStaticMarkup(createElement(DashboardHeader, { data: { generatedAt: value }, mode: 'light' }));
  expect(html).toContain('Data unavailable');
  expect(html).not.toContain('<time');
});
