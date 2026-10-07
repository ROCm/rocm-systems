import { expect } from '@playwright/test';

// Isolate ECharts access here; user interactions still use real pointer/keyboard
// events. Wait for options/animation rather than sleeping after a render.
export async function readChart(chart, extract, argument = null) {
  let extracted;
  await expect.poll(async () => {
    const result = await chart.evaluate((element, { source, payload }) => {
      const fiberKey = Object.keys(element).find((key) => key.startsWith('__reactFiber'));
      let fiber = element[fiberKey];
      while (fiber && typeof fiber.stateNode?.getEchartsInstance !== 'function') fiber = fiber.return;
      if (!fiber) return { ready: false };
      const instance = fiber.stateNode.getEchartsInstance();
      const option = instance.getOption();
      if (!option || instance.__pendingUpdate || !instance.getZr().animation.isFinished()
        || !option.xAxis?.length || !option.yAxis?.length || !Array.isArray(option.series)) return { ready: false };
      return { ready: true, value: new Function(`return (${source});`)()(instance, payload) };
    }, { source: extract.toString(), payload: argument });
    if (result.ready) extracted = result.value;
    return result.ready;
  }, { message: 'ECharts should settle its options', intervals: [16, 32, 50, 100, 200] }).toBe(true);
  return extracted;
}

export function selectedRunIds(chart) {
  return readChart(chart, (instance) => [...new Set(instance.getOption().series
    .filter((series) => series.name.endsWith('selected points'))
    .flatMap((series) => series.data.flatMap((point) => point?.record ? [point.record.run.runId] : [])))].sort());
}

export async function clickLastCompletedChartPoint(chart) {
  await chart.scrollIntoViewIfNeeded();
  let previous;
  let position;
  await expect.poll(async () => {
    position = await readChart(chart, (instance) => {
      const option = instance.getOption();
      const seriesIndex = option.series.findIndex((series) => series.type === 'line'
        && series.data.some((point) => point?.record && Number.isFinite(point.value)));
      const series = option.series[seriesIndex];
      const index = series.data.findLastIndex((point) => point?.record && Number.isFinite(point.value));
      return instance.convertToPixel({ seriesIndex }, [option.xAxis[0].data[index], series.data[index].value]);
    });
    const stable = previous && Math.abs(previous[0] - position[0]) < 0.5 && Math.abs(previous[1] - position[1]) < 0.5;
    previous = position;
    return Boolean(stable);
  }).toBe(true);
  await chart.click({ position: { x: position[0], y: position[1] } });
}
