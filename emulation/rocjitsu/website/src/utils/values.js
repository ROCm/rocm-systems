export function hasDisplayValue(value) {
  if (value == null) return false;
  return typeof value !== 'string' || value.trim().length > 0;
}
