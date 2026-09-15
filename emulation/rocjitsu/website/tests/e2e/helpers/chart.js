import { expect } from '@playwright/test';

// Every chart assertion needs the ECharts instance behind a rendered container. Walking the React
// fiber is the only way to reach it from a locator, so the traversal lives here once and callers
// pass a plain extractor that receives the instance.
export function readChart(chart, extract, argument = null) {
  return chart.evaluate((element, { source, payload }) => {
    const fiberKey = Object.keys(element).find((key) => key.startsWith('__reactFiber'));
    let fiber = element[fiberKey];
    while (fiber && typeof fiber.stateNode?.getEchartsInstance !== 'function') fiber = fiber.return;
    if (!fiber) throw new Error('No ECharts instance is mounted for this locator');
    return new Function(`return (${source});`)()(fiber.stateNode.getEchartsInstance(), payload);
  }, { source: extract.toString(), payload: argument });
}

export function chartOption(chart, extract, argument = null) {
  return readChart(chart, extract, argument);
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

async function clickChartPosition(chart, extract, argument = null) {
  const position = await settledPosition(chart, extract, argument);
  await chart.click({ position: { x: position[0], y: position[1] } });
}

export async function clickCompletedChartPoint(chart, completedOffset = 0) {
  await clickChartPosition(chart, (instance, offset) => {
    const option = instance.getOption();
    const points = option.series[0].data;
    let index = points.length - 1;
    let remaining = offset;
    while (index >= 0) {
      if (points[index] && remaining === 0) break;
      if (points[index]) remaining -= 1;
      index -= 1;
    }
    return instance.convertToPixel(
      { seriesIndex: 0 },
      [option.xAxis[0].data[index], points[index].value],
    );
  }, completedOffset);
}

export async function clickLastCompletedChartPoint(chart) {
  await clickCompletedChartPoint(chart);
}

export async function clickLastStatusChartPoint(chart) {
  await clickChartPosition(chart, (instance) => {
    const option = instance.getOption();
    const seriesIndex = option.series.findIndex((series) => series.name === 'Failed or timed-out test results');
    return instance.convertToPixel({ seriesIndex }, option.series[seriesIndex].data.at(-1).value);
  });
}

export async function clickAggregateIncompletePoint(chart, commitSha) {
  await clickChartPosition(chart, (instance, sha) => {
    const option = instance.getOption();
    const seriesIndex = option.series.findIndex((series) => series.name === 'Incomplete aggregate results');
    const point = option.series[seriesIndex].data.find((candidate) => (
      candidate.run?.provenance?.rocjitsuCommitSha?.startsWith(sha)
    ));
    return instance.convertToPixel({ seriesIndex }, point.value);
  }, commitSha);
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
    .filter((series) => series.name.endsWith('selected points'))
    .flatMap((series) => series.data)
    .filter(Boolean)
    .map((point) => point.record.run.runId))].length);
}

export function dispatchZoom(chart, start, end) {
  return readChart(chart, (instance, range) => {
    instance.dispatchAction({ type: 'dataZoom', start: range.start, end: range.end });
    return null;
  }, { start, end });
}
