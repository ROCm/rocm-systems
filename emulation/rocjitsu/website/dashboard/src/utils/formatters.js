const shortDateFormatter = new Intl.DateTimeFormat(undefined, {
  month: 'short',
  day: 'numeric',
  timeZone: 'UTC',
});

const fullDateFormatter = new Intl.DateTimeFormat(undefined, {
  year: 'numeric',
  month: 'short',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
  timeZone: 'UTC',
  timeZoneName: 'short',
});

export const formatShortDate = (timestamp) => typeof timestamp === 'string' && Number.isFinite(Date.parse(timestamp)) ? shortDateFormatter.format(new Date(timestamp)) : '-';
export const formatFullDate = (timestamp) => typeof timestamp === 'string' && Number.isFinite(Date.parse(timestamp)) ? fullDateFormatter.format(new Date(timestamp)) : '-';

const runTimeFormatter = new Intl.DateTimeFormat(undefined, {
  year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  hour12: false, timeZone: 'UTC',
});

export function formatRunTime(timestamp) {
  return typeof timestamp === 'string' && Number.isFinite(Date.parse(timestamp))
    ? `${runTimeFormatter.format(new Date(timestamp))} UTC` : '—';
}

const publicationDateFormatter = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
const publicationTimeFormatter = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZone: 'UTC' });

export function formatPublicationTime(timestamp) {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : null;
  return date && Number.isFinite(date.getTime())
    ? `${publicationDateFormatter.format(date)} · ${publicationTimeFormatter.format(date)} UTC` : null;
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
