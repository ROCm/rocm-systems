import { hasDisplayValue } from '../utils/values';

function displayValue(value) {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'string' || typeof value === 'number') return String(value);
  return null;
}

function displayKey(key) {
  const words = String(key)
    .replaceAll(/[_-]+/g, ' ')
    .replaceAll(/([a-z0-9])([A-Z])/g, '$1 $2')
    .toLocaleLowerCase();
  return words.replace(/^./, (character) => character.toUpperCase());
}

export function problemDetails(problem = {}) {
  return Object.entries(problem).flatMap(([key, value]) => {
    const normalizedKey = hasDisplayValue(key) ? String(key).trim() : '';
    const normalizedValue = displayValue(value);
    if (!normalizedKey || !hasDisplayValue(normalizedValue)) return [];
    return [{ key: normalizedKey, label: displayKey(normalizedKey), value: normalizedValue }];
  });
}
