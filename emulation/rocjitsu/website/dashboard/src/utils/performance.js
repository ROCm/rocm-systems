export { classifyDurationChange, DURATION_CHANGE_BAND_PERCENT } from '../../../../scripts/benchmark-data/comparison.mjs';

export function changeTone(changeState) {
  if (changeState === 'slower') return 'error';
  if (changeState === 'faster') return 'success';
  return 'neutral';
}
