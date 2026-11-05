// Calendar keys identify a day in the viewer's timezone, not a UTC instant.
export function localDay(timestamp) {
  const date = new Date(timestamp);
  if (timestamp == null || !Number.isFinite(date.getTime())) return null;
  const calendar = new Date(0);
  calendar.setUTCFullYear(date.getFullYear(), date.getMonth(), date.getDate());
  return calendar.toISOString().split('T')[0];
}

export const calendarDate = (day) => new Date(`${day}T12:00:00`);
