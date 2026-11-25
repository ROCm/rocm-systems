import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, expect, test, vi } from 'vitest';
import { normalizeEvent } from 'zrender/lib/core/event.js';
import Chart from '../../src/components/shared/Chart.jsx';

const capture = vi.hoisted(() => ({ props: null, cleanups: [] }));
vi.mock('echarts-for-react/esm/core', () => ({ default: (props) => { capture.props = props; return createElement('div'); } }));
vi.mock('react', async (load) => ({ ...await load(), useLayoutEffect: (effect) => { capture.cleanups.push(effect()); } }));
beforeEach(() => { capture.props = null; capture.cleanups = []; });

function renderer(scale = 0.8) {
  const viewport = new EventTarget();
  viewport.bounds = { left: 100, top: 60, width: 100 * scale, height: 50 * scale };
  viewport.getBoundingClientRect = () => viewport.bounds;
  const chart = { isDisposed: () => false, getWidth: () => 100, getHeight: () => 50,
    getZr: () => ({ painter: { getViewportRoot: () => viewport } }) };
  return { chart, viewport };
}

function ready(chart, onReady = vi.fn()) {
  renderToStaticMarkup(createElement(Chart, { option: {}, onReady }));
  capture.props.onChartReady(chart);
  return onReady;
}

function fire(viewport, type = 'click', fields = {}) {
  const event = Object.assign(new Event(type), { clientX: 132, clientY: 76, offsetX: 32, offsetY: 16, button: 0, ...fields });
  viewport.dispatchEvent(event);
  // The real installed renderer normalizes after chart-owned capture.
  normalizeEvent(viewport, event);
  return event;
}

test.each([0.8, 1, 1.25])('native chart input uses current measured scale %s before renderer hit testing', (scale) => {
  const { chart, viewport } = renderer(scale);
  ready(chart);
  const event = fire(viewport, 'click', { clientX: 100 + 40 * scale, clientY: 60 + 20 * scale, offsetX: 40 * scale, offsetY: 20 * scale });
  expect([event.zrX, event.zrY]).toEqual([40, 20]);
  expect([event.clientX, event.clientY, event.offsetX, event.offsetY]).toEqual([100 + 40 * scale, 60 + 20 * scale, 40 * scale, 20 * scale]);
  capture.cleanups.forEach((cleanup) => cleanup?.());
});

test.each(['mousedown', 'mouseup', 'mousemove', 'mouseout', 'pointerdown', 'pointerup', 'pointermove', 'pointerout', 'click', 'dblclick', 'contextmenu'])('mouse/pen %s retains metadata and consistent logical coordinates', (type) => {
  const { chart, viewport } = renderer();
  ready(chart);
  const event = fire(viewport, type, { pointerType: 'pen', button: 2, zrEventControl: 'no_globalout' });
  expect([event.zrX, event.zrY]).toEqual([40, 20]);
  expect(event.which).toBe(3);
  expect(event.zrDelta).toBe(-0);
  expect(event.pointerType).toBe('pen');
  expect(event.zrEventControl).toBe('no_globalout');
  capture.cleanups.forEach((cleanup) => cleanup?.());
});

test.each(['wheel', 'mousewheel'])('%s uses actual installed wheel metadata before coordinate correction', (type) => {
  const { chart, viewport } = renderer();
  ready(chart);
  const event = fire(viewport, type, { deltaX: 0, deltaY: 120 });
  expect(event.zrDelta).toBe(-3);
  expect([event.zrX, event.zrY]).toEqual([40, 20]);
  expect(event.deltaY).toBe(120);
  capture.cleanups.forEach((cleanup) => cleanup?.());
});

test.each(['touchstart', 'touchmove', 'touchend'])('%s preserves contact fields and existing touch flags', (type) => {
  const { chart, viewport } = renderer();
  ready(chart);
  const touch = { clientX: 132, clientY: 76, offsetX: 32, offsetY: 16 };
  const event = fire(viewport, type, { targetTouches: type === 'touchend' ? [] : [touch], changedTouches: [touch], zrByTouch: true });
  expect([event.zrX, event.zrY]).toEqual([40, 20]);
  expect(event.zrDelta).toBeUndefined();
  expect(event.zrByTouch).toBe(true);
  expect(touch).toEqual({ clientX: 132, clientY: 76, offsetX: 32, offsetY: 16 });
  capture.cleanups.forEach((cleanup) => cleanup?.());
});

test('current geometry and logical dimensions are read again after resize', () => {
  const { chart, viewport } = renderer();
  ready(chart);
  viewport.bounds = { left: 20, top: 10, width: 125, height: 62.5 };
  chart.getWidth = () => 200;
  chart.getHeight = () => 100;
  expect([fire(viewport, 'click', { clientX: 70, clientY: 35 }).zrX, fire(viewport, 'click', { clientX: 70, clientY: 35 }).zrY]).toEqual([80, 40]);
  capture.cleanups.forEach((cleanup) => cleanup?.());
});

test.each([{ field: 'width', value: 0 }, { field: 'height', value: -1 }, { field: 'left', value: NaN }, { field: 'top', value: Infinity }])('invalid viewport $field=$value does not seed coordinates', ({ field, value }) => {
  const { chart, viewport } = renderer();
  ready(chart);
  viewport.bounds[field] = value;
  const event = Object.assign(new Event('click'), { clientX: 132, clientY: 76, offsetX: 32, offsetY: 16 });
  expect(() => viewport.dispatchEvent(event)).not.toThrow();
  expect(event.zrX).toBeUndefined();
  capture.cleanups.forEach((cleanup) => cleanup?.());
});

test('nonfinite logical dimensions or missing touch/client coordinates are untouched', () => {
  const { chart, viewport } = renderer();
  ready(chart);
  chart.getWidth = () => Infinity;
  const invalid = Object.assign(new Event('click'), { clientX: 132, clientY: 76 });
  viewport.dispatchEvent(invalid);
  expect(invalid.zrX).toBeUndefined();
  chart.getWidth = () => 100;
  for (const event of [new Event('click'), new Event('touchend')]) {
    viewport.dispatchEvent(event);
    expect(event.zrX).toBeUndefined();
  }
  capture.cleanups.forEach((cleanup) => cleanup?.());
});

test('an already-normalized or reused native event is not corrected twice', () => {
  const { chart, viewport } = renderer();
  ready(chart);
  const prior = fire(viewport, 'click', { zrX: 17, zrY: 19, zrDelta: 2, which: 1 });
  expect([prior.zrX, prior.zrY, prior.zrDelta, prior.which]).toEqual([17, 19, 2, 1]);
  const event = fire(viewport);
  viewport.bounds.width = 200;
  viewport.dispatchEvent(event);
  expect([event.zrX, event.zrY]).toEqual([40, 20]);
  capture.cleanups.forEach((cleanup) => cleanup?.());
});

test('ready replacement and layout cleanup remove capture listeners and forward callbacks', () => {
  const first = renderer(); const second = renderer();
  const removeFirst = vi.spyOn(first.viewport, 'removeEventListener');
  const addFirst = vi.spyOn(first.viewport, 'addEventListener');
  const removeSecond = vi.spyOn(second.viewport, 'removeEventListener');
  const onReady = ready(first.chart);
  const count = addFirst.mock.calls.length;
  expect(count).toBeGreaterThan(0);
  expect(addFirst.mock.calls.every(([, , options]) => options.capture && options.passive)).toBe(true);
  capture.props.onChartReady(first.chart);
  expect(removeFirst).toHaveBeenCalledTimes(count);
  expect(onReady).toHaveBeenNthCalledWith(2, first.chart);
  capture.props.onChartReady(second.chart);
  expect(removeFirst).toHaveBeenCalledTimes(count * 2);
  expect(fire(first.viewport).zrX).toBe(32);
  expect(fire(second.viewport).zrX).toBe(40);
  expect(onReady).toHaveBeenNthCalledWith(3, second.chart);
  capture.cleanups.forEach((cleanup) => cleanup?.());
  expect(removeSecond).toHaveBeenCalledTimes(count);
  expect(fire(second.viewport).zrX).toBe(32);
});

test('disposed renderer input detaches its listeners without touching the event', () => {
  const { chart, viewport } = renderer();
  const remove = vi.spyOn(viewport, 'removeEventListener');
  ready(chart);
  chart.isDisposed = () => true;
  const event = new Event('click');
  expect(() => viewport.dispatchEvent(event)).not.toThrow();
  expect(event.zrX).toBeUndefined();
  expect(remove.mock.calls.length).toBeGreaterThan(0);
});

test('an already disposed renderer is not inspected, and ready/onEvents forwarding remains unchanged', () => {
  const { chart } = renderer();
  chart.isDisposed = () => true;
  chart.getZr = vi.fn();
  const onEvents = { click: vi.fn() }; const onReady = vi.fn();
  renderToStaticMarkup(createElement(Chart, { option: {}, onEvents, onReady }));
  capture.props.onChartReady(chart);
  expect(chart.getZr).not.toHaveBeenCalled();
  expect(onReady).toHaveBeenCalledWith(chart);
  expect(capture.props.onEvents).toBe(onEvents);
});
