import { expect } from '@playwright/test';

// Every chart assertion needs the ECharts instance behind a rendered container. Walking the React
// fiber is the only way to reach it from a locator, so the traversal lives here once and callers
// pass a plain extractor that receives the instance.
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
      const ready = Boolean(option)
        && !instance.__pendingUpdate
        && instance.getZr().animation.isFinished()
        && Array.isArray(option.xAxis)
        && option.xAxis.length > 0
        && Array.isArray(option.yAxis)
        && option.yAxis.length > 0
        && Array.isArray(option.series);
      if (!ready) return { ready: false };

      return {
        ready: true,
        value: new Function(`return (${source});`)()(instance, payload),
      };
    }, { source: extract.toString(), payload: argument });
    if (result.ready) extracted = result.value;
    return result.ready;
  }, {
    message: 'ECharts should finish applying its option and animation',
    intervals: [16, 32, 50, 100, 200],
  }).toBe(true);
  return extracted;
}

// ECharts animates layout changes, so a pixel read straight after a state change can point at a
// stale position. Sampling until two consecutive reads agree replaces fixed sleeps before clicks.
async function settledPosition(chart, extract, argument = null) {
  let previous = null;
  let settled = null;
  await expect.poll(async () => {
    const current = await readChart(chart, extract, argument);
    const stable = Boolean(previous)
      && Math.abs(previous[0] - current[0]) < 0.5
      && Math.abs(previous[1] - current[1]) < 0.5;
    previous = current;
    settled = current;
    return stable;
  }, { intervals: [50, 100, 100, 200, 300, 500, 500] }).toBe(true);
  return settled;
}

async function clickChartPosition(chart, extract, argument, observeChange) {
  const stateBefore = await observeChange();
  const position = await settledPosition(chart, extract, argument);
  await chart.click({ position: { x: position[0], y: position[1] } });
  await expect.poll(
    observeChange,
    { message: 'chart click should change application state' },
  ).not.toEqual(stateBefore);
}

export async function clickCompletedChartPoint(
  chart,
  completedOffset = 0,
  observeChange = () => selectedRunCount(chart),
) {
  await clickChartPosition(chart, (instance, offset) => {
    const option = instance.getOption();
    const completed = [];
    option.series.forEach((series, seriesIndex) => {
      if (series.type !== 'line' || series.name.includes('missed-data bridge')) return;
      series.data.forEach((point, index) => {
        const value = typeof point === 'number' ? point : point?.value;
        if (point && Number.isFinite(value)) {
          completed.push({ seriesIndex, index, value });
        }
      });
    });
    completed.sort((left, right) => left.index - right.index);
    const chosen = completed.at(-(offset + 1));
    return instance.convertToPixel(
      { seriesIndex: chosen.seriesIndex },
      [option.xAxis[0].data[chosen.index], chosen.value],
    );
  }, completedOffset, observeChange);
}

export async function clickLastCompletedChartPoint(
  chart,
  observeChange = () => selectedRunCount(chart),
) {
  await clickCompletedChartPoint(chart, 0, observeChange);
}

export async function clickLastStatusChartPoint(
  chart,
  observeChange = () => selectedRunCount(chart),
) {
  await clickChartPosition(chart, (instance) => {
    const option = instance.getOption();
    const seriesIndex = option.series.findIndex((series) => series.name === 'Failed or timed-out test results');
    return instance.convertToPixel({ seriesIndex }, option.series[seriesIndex].data.at(-1).value);
  }, null, observeChange);
}

export async function clickAggregateIncompletePoint(
  chart,
  commitSha,
  observeChange = () => selectedRunCount(chart),
) {
  await clickChartPosition(chart, (instance, sha) => {
    const option = instance.getOption();
    const seriesIndex = option.series.findIndex((series) => series.name === 'Incomplete aggregate results');
    const point = option.series[seriesIndex].data.find((candidate) => (
      candidate.run?.provenance?.rocjitsuCommitSha?.startsWith(sha)
    ));
    return instance.convertToPixel({ seriesIndex }, point.value);
  }, commitSha, observeChange);
}

export function chartScale(chart) {
  return readChart(chart, (instance) => {
    const option = instance.getOption();
    return {
      zoom: option.dataZoom.map((item) => ({
        start: Number(item.start.toFixed(4)),
        end: Number(item.end.toFixed(4)),
      })),
      yExtent: instance.getModel().getComponent('yAxis').axis.scale.getExtent()
        .map((value) => Number(value.toFixed(4))),
    };
  });
}

export async function settledChartScale(chart) {
  let previous = null;
  let settled = null;
  await expect.poll(async () => {
    const current = await chartScale(chart);
    const stable = JSON.stringify(previous) === JSON.stringify(current);
    previous = current;
    settled = current;
    return stable;
  }, { intervals: [50, 100, 100, 200, 300, 500] }).toBe(true);
  return settled;
}

export function selectedDotsViewport(chart, selectedSeriesName) {
  return readChart(chart, (instance, seriesName) => {
    const option = instance.getOption();
    const indexes = [...new Set(option.series
      .filter((series) => series.name === seriesName || series.name.endsWith(seriesName))
      .flatMap((series) => series.data.flatMap((point, index) => {
        if (!point) return [];
        if (Array.isArray(point.value)) return [point.value[0]];
        return [index];
      })))];
    const grid = instance.getModel().getComponent('grid')?.coordinateSystem?.getRect();
    const pixels = indexes.map((index) => instance.convertToPixel(
      { xAxisIndex: 0 },
      option.xAxis[0].data[index],
    ));
    const zoom = option.dataZoom[0];
    return {
      indexes,
      pixels,
      startValue: zoom.startValue,
      endValue: zoom.endValue,
      allVisible: Boolean(grid)
        && pixels.every((pixel) => pixel >= grid.x && pixel <= grid.x + grid.width),
    };
  }, selectedSeriesName);
}

export function selectedRunCount(chart) {
  return readChart(chart, (instance) => [...new Set(instance.getOption().series
    .flatMap((series) => series.data
      .filter((point) => point && (
        point.selected
        || series.name === 'Selected run'
        || series.name.endsWith('selected points')
      ))
      .map((point) => point.record?.run?.runId ?? point.run?.runId))
    .filter(Boolean))].length);
}

export function dispatchZoom(chart, start, end) {
  return readChart(chart, (instance, range) => {
    instance.dispatchAction({ type: 'dataZoom', start: range.start, end: range.end });
    return null;
  }, { start, end });
}
