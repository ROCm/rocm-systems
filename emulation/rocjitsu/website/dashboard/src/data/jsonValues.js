const MAX_TIMING_SECONDS = 4503599627370495;
export const isSupportedNumber = (value) => typeof value === 'number' && Number.isFinite(value)
  && Math.abs(value) <= Number.MAX_SAFE_INTEGER;
export const isTimingNumber = (value) => isSupportedNumber(value) && value > 0 && value <= MAX_TIMING_SECONDS;

export const isJsonObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

export function isJsonValue(value) {
  return value === null || typeof value === 'string' || typeof value === 'boolean'
    || isSupportedNumber(value)
    || Array.isArray(value) && value.every(isJsonValue)
    || isJsonObject(value) && Object.entries(value).every(([key, item]) => Boolean(key.trim()) && isJsonValue(item));
}

export function typedJsonIdentity(value) {
  if (Array.isArray(value)) return `[${value.map(typedJsonIdentity).join(',')}]`;
  if (isJsonObject(value)) {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${typedJsonIdentity(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}
