import { isValidTimestamp } from './timestamp.js';

const shortDateFormatter = new Intl.DateTimeFormat(undefined, {
  month: 'short',
  day: 'numeric',
  timeZone: 'UTC',
});

export const formatShortDate = (timestamp) => isValidTimestamp(timestamp) ? shortDateFormatter.format(new Date(timestamp)) : '-';
const runTimeFormatter = new Intl.DateTimeFormat(undefined, {
  year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  hour12: false, timeZoneName: 'short',
});

export function formatRunTime(timestamp) {
  return isValidTimestamp(timestamp)
    ? runTimeFormatter.format(new Date(timestamp)) : '—';
}

// Same text as formatRunTime; differs only in showing '-' rather than '—' for a missing or invalid time.
export const formatFullDate = (timestamp) => (isValidTimestamp(timestamp) ? formatRunTime(timestamp) : '-');

const compactRunTimeFormatter = new Intl.DateTimeFormat(undefined, {
  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  hour12: false,
});

export function formatCompactRunTime(timestamp) {
  return isValidTimestamp(timestamp)
    ? compactRunTimeFormatter.format(new Date(timestamp)) : '-';
}

const publicationDateFormatter = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' });
const publicationTimeFormatter = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZoneName: 'short' });

export function formatPublicationTime(timestamp) {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : null;
  return date && Number.isFinite(date.getTime())
    ? `${publicationDateFormatter.format(date)} · ${publicationTimeFormatter.format(date)}` : null;
}

export function formatMetadataValue(value) {
  if (value == null) return '—';
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

export function formatDuration(value) {
  if (!Number.isFinite(value)) return '—';
  if (value >= 60) {
    const seconds = Number((value % 60).toFixed(1));
    const carry = seconds === 60;
    return `${Math.floor(value / 60) + Number(carry)}m ${(carry ? 0 : seconds).toFixed(1)}s`;
  }
  return `${value.toFixed(value < 10 ? 2 : 1)}s`;
}

export function formatPercent(value, withSign = true) {
  if (!Number.isFinite(value)) return '—';
  const sign = withSign && value > 0 ? '+' : '';
  return `${sign}${value.toFixed(1)}%`;
}

export const shortSha = (run) => typeof run?.provenance?.rocjitsuCommitSha === 'string' && run.provenance.rocjitsuCommitSha ? run.provenance.rocjitsuCommitSha.slice(0, 8) : '-';

const htmlEscapes = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

export const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => htmlEscapes[character]);
