function numericPointValue(point) {
  const value = typeof point === 'number' ? point : point?.value;
  return Number.isFinite(value) ? value : null;
}

export function chartGapPresentation(points = []) {
  const observedValues = points.map(numericPointValue);
  const previousIndexes = [];
  const nextIndexes = [];

  let previousIndex = -1;
  observedValues.forEach((value, index) => {
    previousIndexes[index] = previousIndex;
    if (Number.isFinite(value)) previousIndex = index;
  });

  let nextIndex = -1;
  for (let index = observedValues.length - 1; index >= 0; index -= 1) {
    nextIndexes[index] = nextIndex;
    if (Number.isFinite(observedValues[index])) nextIndex = index;
  }

  const estimatedValues = observedValues.map((value, index) => {
    if (Number.isFinite(value)) return value;
    const leftIndex = previousIndexes[index];
    const rightIndex = nextIndexes[index];
    if (leftIndex >= 0 && rightIndex >= 0) {
      const progress = (index - leftIndex) / (rightIndex - leftIndex);
      return observedValues[leftIndex]
        + ((observedValues[rightIndex] - observedValues[leftIndex]) * progress);
    }
    if (leftIndex >= 0) return observedValues[leftIndex];
    if (rightIndex >= 0) return observedValues[rightIndex];
    return null;
  });

  const segments = [];
  for (let index = 0; index < observedValues.length;) {
    if (Number.isFinite(observedValues[index])) {
      index += 1;
      continue;
    }

    const gapStart = index;
    while (index < observedValues.length && !Number.isFinite(observedValues[index])) index += 1;
    const gapEnd = index - 1;
    const segmentStart = gapStart > 0 ? gapStart - 1 : gapStart;
    const segmentEnd = index < observedValues.length ? index : gapEnd;
    const data = Array(observedValues.length).fill(null);
    for (let pointIndex = segmentStart; pointIndex <= segmentEnd; pointIndex += 1) {
      data[pointIndex] = estimatedValues[pointIndex];
    }
    if (data.filter(Number.isFinite).length >= 2) segments.push(data);
  }

  return { estimatedValues, segments };
}
