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

export function formatRelativeTime(timestamp) {
  const seconds = (Date.parse(timestamp) - Date.now()) / 1000;
  if (!Number.isFinite(seconds)) return 'unavailable';
  const units = [
    ['year', 31_536_000],
    ['month', 2_592_000],
    ['week', 604_800],
    ['day', 86_400],
    ['hour', 3_600],
    ['minute', 60],
    ['second', 1],
  ];
  const [unit, size] = units.find(([, unitSize]) => Math.abs(seconds) >= unitSize) ?? units.at(-1);
  return new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' }).format(Math.round(seconds / size), unit);
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
