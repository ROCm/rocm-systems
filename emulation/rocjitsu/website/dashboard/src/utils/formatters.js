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

export const formatShortDate = (timestamp) => shortDateFormatter.format(new Date(timestamp));
export const formatFullDate = (timestamp) => fullDateFormatter.format(new Date(timestamp));

export function formatMetadataValue(value) {
  if (value == null) return 'Not provided';
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

export function formatDuration(value) {
  if (!Number.isFinite(value)) return '—';
  if (value >= 60) return `${Math.floor(value / 60)}m ${(value % 60).toFixed(1)}s`;
  return `${value.toFixed(value < 10 ? 2 : 1)}s`;
}

export function formatPercent(value, withSign = true) {
  if (!Number.isFinite(value)) return '—';
  const sign = withSign && value > 0 ? '+' : '';
  return `${sign}${value.toFixed(1)}%`;
}

export const shortSha = (run) => run?.provenance?.rocjitsuCommitSha?.slice(0, 8) ?? '—';

const htmlEscapes = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

export const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => htmlEscapes[character]);
