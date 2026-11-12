import { fontSizes } from '../theme/typography';
import { formatShortDate } from './formatters';

// Commit-day x axis shared by Overview's trend and Benchmark charts.
// `slots`, `dayKeys` and `axisMax` come from the day-slot projection; `firstSlot` is the
// first slot that carries data (1M/3M always start at the first day).
export function dayAxisBounds({ range, slots, dayKeys, axisMax }, firstSlot) {
  const weekly = range === '1W';
  const firstX = range === '1M' || range === '3M' ? 0 : firstSlot != null ? slots[firstSlot].x : 0;
  const lastX = Math.max(firstX + 1, weekly ? Math.max(1, axisMax - 0.5, ...slots.map((slot) => slot.x)) : axisMax);
  const dateStep = Math.max(1, Math.ceil((lastX - firstX) / 6));
  return { weekly, firstX, lastX, dateStep, dayKeys };
}

// `leftPad` widens the plot to the left (in day units) so the first point clears the y axis.
export function dayXAxis({ weekly, firstX, lastX, dateStep, dayKeys }, theme, { leftPad = 0 } = {}) {
  return {
    type: 'value', min: firstX - leftPad, max: lastX,
    axisTick: { show: false }, splitLine: { show: false },
    axisLabel: {
      color: theme.palette.text.secondary, fontSize: fontSizes.caption, hideOverlap: true,
      customValues: weekly ? dayKeys.map((_, index) => index + 0.5)
        : Array.from({ length: Math.floor((lastX - firstX) / dateStep) + 1 }, (_, index) => firstX + index * dateStep),
      formatter: (x) => {
        const day = dayKeys[weekly ? Math.floor(x) : Math.round(x)];
        return day ? formatShortDate(`${day}T12:00:00Z`) : '';
      },
    },
    axisLine: { show: true, onZero: false, lineStyle: { type: 'solid', color: theme.palette.dashboard?.border ?? theme.palette.divider } },
  };
}
