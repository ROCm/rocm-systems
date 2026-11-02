export const DURATION_CHANGE_BAND_PERCENT = 3;

export function classifyDurationChange(value, tolerance = DURATION_CHANGE_BAND_PERCENT) {
  if (!Number.isFinite(value)) return 'unavailable';
  if (value > tolerance) return 'slower';
  if (value < -tolerance) return 'faster';
  return 'neutral';
}

export function changeTone(changeState) {
  if (changeState === 'slower') return 'error';
  if (changeState === 'faster') return 'success';
  return 'neutral';
}
