import { createReactRootSurface } from '../helpers/react-root-surface.js';
import { act, Children, createElement, isValidElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, test, vi } from 'vitest';
import DurationHistory from '../../src/components/overview/DurationHistory.jsx';

const roots = [];
afterEach(() => { for (const root of roots.splice(0)) act(() => root.unmount()); vi.unstubAllGlobals(); });
function nodes(tree) {
  const result = [];
  function visit(element) {
    if (!isValidElement(element)) return;
    result.push(element);
    Children.forEach(element.props.children, visit);
  }
  Children.forEach(tree, visit);
  return result;
}
function mountTrend(overrides = {}) {
  const { container } = createReactRootSurface();

  const history = {
    series: [{ data: [120, 100], baseline: 120 }],
    slots: [0, 1].map((x) => ({ x, run: { runId: `fictional-${x}`, timestamp: '2026-10-01T12:00:00Z', commitTimestamp: '2026-10-01T11:00:00Z' } })),
    dayKeys: ['2026-10-01', '2026-10-02'], axisMax: 1, currentDuration: 100, durationDelta: -16.67,
    ...overrides,
  };
  let tree, renders = 0;
  function Probe() { renders++; tree = DurationHistory({ history, range: 'ALL', onRangeChange() {} }); return null; }
  const root = createRoot(container); roots.push(root);
  act(() => root.render(createElement(Probe)));
  return {
    renders: () => renders,
    chart: () => nodes(tree).find((node) => node.props.onEvents),
    group: () => nodes(tree).find((node) => node.props['aria-label'] === 'Inspect performance trend'),
    status: () => nodes(tree).find((node) => node.props.role === 'status'),
    anchors: () => nodes(tree).find((node) => node.props['aria-label'] === 'Selected trend estimate anchors'),
  };
}

test.each(['touch', 'keyboard'])('%s trend inspection survives an empty ECharts axis-pointer release event', (source) => {
  const probe = mountTrend();
  if (source === 'touch') {
    act(() => probe.chart().props.onEvents.finished({}, { isDisposed: () => false, containPixel: () => true, convertFromPixel: () => [1, 100] }));
    act(() => probe.group().props.onPointerDown({ pointerType: 'touch', clientX: 100, clientY: 50, currentTarget: { getBoundingClientRect: () => ({ left: 0, top: 0 }) } }));
  } else act(() => probe.group().props.onKeyDown({ key: 'End', preventDefault() {} }));
  expect(probe.status()).toBeDefined();
  // ECharts emits this when the tooltip/axis pointer hides after touch release.
  act(() => probe.chart().props.onEvents.updateAxisPointer({ axesInfo: [] }));
  expect(probe.status()).toBeDefined();
  act(() => probe.chart().props.onEvents.globalout());
  act(() => probe.group().props.onPointerLeave());
  expect(probe.status()).toBeDefined();
  act(() => probe.group().props.onKeyDown({ key: 'Escape', preventDefault() {} }));
  expect(probe.status()).toBeUndefined();
});

test('touch trend opts into synchronous ECharts updates before follow-up pointer events', () => {
  const probe = mountTrend();
  expect(probe.chart().props.lazyUpdate).toBe(false);
  expect(probe.chart().props.height).toBe(320);
  expect(probe.chart().props.option.grid.left).toBe(40);
  expect(probe.group().props.sx).toMatchObject({ mt: 2, minHeight: 320 });
});

test('repeated identical axis-pointer events do not restart the same trend inspection render', () => {
  const probe = mountTrend();
  act(() => probe.chart().props.onEvents.updateAxisPointer({ axesInfo: [{ value: 1 }] }));
  const before = probe.renders();
  // A replacement chart option can re-emit its existing axis pointer.
  for (let i = 0; i < 3; i++) act(() => probe.chart().props.onEvents.updateAxisPointer({ axesInfo: [{ value: 1 }] }));
  expect(probe.renders()).toBeLessThanOrEqual(before + 1);
  act(() => probe.chart().props.onEvents.updateAxisPointer({ axesInfo: [{ value: 0 }] }));
  expect(probe.renders()).toBeGreaterThan(before);
});

test('cursor sweeps keep the complete trend model and guides stable', () => {
  const probe = mountTrend();
  const initial = probe.chart().props.option;
  const handlers = probe.chart().props.onEvents;
  act(() => probe.chart().props.onEvents.updateAxisPointer({ axesInfo: [{ value: 0 }] }));
  expect(probe.chart().props.option).toBe(initial);
  act(() => probe.chart().props.onEvents.updateAxisPointer({ axesInfo: [{ value: 1 }] }));
  expect(probe.chart().props.option).toBe(initial);
  expect(initial.tooltip.axisPointer.animation).toBe(false);
  expect(probe.chart().props.onEvents).toBe(handlers);
  expect(initial.tooltip.transitionDuration).toBe(0);
  expect(initial.series[0].markLine.data).toEqual([{ yAxis: 2 }]);
  act(() => probe.group().props.onPointerLeave());
  expect(probe.chart().props.option).toBe(initial);
});

test('keyboard inspection uses native pointer actions and explicitly clears its guide', () => {
  const probe = mountTrend();
  const dispatchAction = vi.fn();
  act(() => probe.chart().props.onEvents.finished({}, { isDisposed: () => false, dispatchAction }));
  act(() => probe.group().props.onKeyDown({ key: 'End', preventDefault() {} }));
  expect(dispatchAction).toHaveBeenCalledWith({ type: 'showTip', seriesIndex: 0, dataIndex: 1 });
  const calls = dispatchAction.mock.calls.length;
  act(() => probe.chart().props.onEvents.updateAxisPointer({ axesInfo: [{ value: 1 }] }));
  expect(dispatchAction).toHaveBeenCalledTimes(calls);
  act(() => probe.group().props.onKeyDown({ key: 'Escape', preventDefault() {} }));
  expect(dispatchAction).toHaveBeenCalledWith({ type: 'hideTip' });
  expect(dispatchAction).toHaveBeenCalledWith({ type: 'updateAxisPointer', currTrigger: 'leave' });
});

test('mouse inspection and anchor disclosure remain reachable after leaving the graph', () => {
  const probe = mountTrend({ estimates: [{ estimatedRunId: 'fictional-1', runId: 'fictional-anchor', target: 'gfx1250', mode: 'ST', logicalTestId: 'added-workload', durationSeconds: 4, timestamp: '2026-10-01T12:00:00Z' }] });
  const dispatchAction = vi.fn();
  act(() => probe.chart().props.onEvents.finished({}, { isDisposed: () => false, dispatchAction }));
  act(() => probe.chart().props.onEvents.updateAxisPointer({ axesInfo: [{ value: 1 }] }));
  expect(probe.status()).toBeDefined();
  expect(probe.anchors()).toBeDefined();
  act(() => probe.group().props.onPointerLeave());
  expect(nodes(probe.group()).some((node) => node.props.role === 'status')).toBe(false);
  expect(probe.status().props.sx).toMatchObject({ width: '1px', height: '1px', overflow: 'hidden' });
  act(() => probe.chart().props.onEvents.globalout());
  expect(probe.status()).toBeDefined();
  expect(probe.anchors()).toBeDefined();
  expect(dispatchAction).toHaveBeenCalledWith({ type: 'hideTip' });
  expect(dispatchAction).toHaveBeenCalledWith({ type: 'updateAxisPointer', currTrigger: 'leave' });
  act(() => probe.group().props.onKeyDown({ key: 'Escape', preventDefault() {} }));
  expect(probe.status()).toBeUndefined();
  expect(probe.anchors()).toBeUndefined();
});

test.each([null, 110])('missing slots do not select adjacent measurements or draw guides (value %s)', (value) => {
  const probe = mountTrend({
    series: [{ data: [120, value, 100], baseline: 120 }], axisMax: 2,
    slots: [0, 1, 2].map((x) => ({ x, run: x === 1 && value !== null ? null : { runId: `fictional-${x}`, timestamp: '2026-10-01T12:00:00Z' } })),
  });
  const dispatchAction = vi.fn();
  act(() => probe.chart().props.onEvents.finished({}, { isDisposed: () => false, dispatchAction, containPixel: () => true, convertFromPixel: () => [1, 100] }));
  expect(probe.chart().props.option.tooltip.axisPointer.snap).toBe(false);
  act(() => probe.chart().props.onEvents.updateAxisPointer({ axesInfo: [{ value: 1 }] }));
  expect(probe.status()).toBeUndefined();
  expect(dispatchAction).toHaveBeenCalledWith({ type: 'updateAxisPointer', currTrigger: 'leave' });
  act(() => probe.chart().props.onEvents.updateAxisPointer({ axesInfo: [{ value: 2 }] }));
  const renders = probe.renders();
  act(() => probe.chart().props.onEvents.updateAxisPointer({ axesInfo: [{ value: 1 }] }));
  act(() => probe.chart().props.onEvents.click({ componentType: 'series', dataIndex: 1 }));
  act(() => probe.group().props.onPointerDown({ pointerType: 'touch', clientX: 100, clientY: 50, currentTarget: { getBoundingClientRect: () => ({ left: 0, top: 0 }) } }));
  expect(probe.renders()).toBe(renders);
  expect(probe.status()).toBeDefined();
});
