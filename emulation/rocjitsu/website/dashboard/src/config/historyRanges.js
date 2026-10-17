// Public URL/UI choices are distinct from Overview's compatibility ranges.
export const historyRanges = [
  { id: '1W', days: 7, label: 'Trailing 7 days', period: 'past week' },
  { id: '1M', days: 30, label: 'Trailing 30 days', period: 'past month' },
  { id: '3M', days: 90, label: 'Trailing 90 days', period: 'past 3 months' },
  { id: 'ALL', days: null, label: 'All available history', period: 'all available history' },
];

export const historyRangeById = new Map(historyRanges.map((range) => [range.id, range]));
const compatibilityRanges = new Map([
  ['1D', { id: '1D', days: 1 }],
  ['6M', { id: '6M', days: 180 }],
  ['YTD', { id: 'YTD', days: null }],
]);

export function resolveHistoryRange(value, { compatibility = false } = {}) {
  return historyRangeById.get(value)
    ?? (compatibility ? compatibilityRanges.get(value) : null)
    ?? historyRangeById.get('ALL');
}
