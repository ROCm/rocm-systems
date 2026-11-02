function niceStep(value) {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const exponent = Math.floor(Math.log10(value));
  const magnitude = 10 ** exponent;
  const normalized = value / magnitude;
  const multiplier = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.5 ? 2.5 : normalized <= 5 ? 5 : 10;
  return multiplier * magnitude;
}

function rounded(value) {
  return Number(value.toPrecision(12));
}

export function durationAxisBounds(values) {
  const durations = values.filter((value) => Number.isFinite(value) && value >= 0);
  if (durations.length === 0) return {};

  const dataMinimum = Math.min(...durations);
  const dataMaximum = Math.max(...durations);
  const midpoint = (dataMinimum + dataMaximum) / 2;
  const observedSpan = dataMaximum - dataMinimum;
  const paddedObservedSpan = observedSpan * 1.3;
  const contextualSpan = Math.max(dataMaximum, midpoint) * 0.35;
  const desiredSpan = Math.max(paddedObservedSpan, contextualSpan, Number.EPSILON);
  const rawMinimum = Math.max(0, midpoint - desiredSpan / 2);
  const rawMaximum = midpoint + desiredSpan / 2;
  const step = niceStep(desiredSpan / 5);

  let min = Math.max(0, Math.floor(rawMinimum / step) * step);
  let max = Math.ceil(rawMaximum / step) * step;
  if (max <= dataMaximum) max += step;
  if (min >= dataMinimum && min > 0) min = Math.max(0, min - step);

  return { min: rounded(min), max: rounded(max) };
}
