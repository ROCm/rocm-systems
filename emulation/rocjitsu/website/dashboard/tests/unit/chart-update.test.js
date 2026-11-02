import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import Chart from '../../src/components/shared/Chart.jsx';

const capture = vi.hoisted(() => ({ props: null }));
vi.mock('echarts-for-react/esm/core', () => ({ default: (props) => { capture.props = props; return createElement('div'); } }));

test('all shared HTML tooltips use the pixel-aligned paint path and preserve caller content', () => {
  const formatter = () => 'Candidate / Baseline';
  renderToStaticMarkup(createElement(Chart, { option: { tooltip: { trigger: 'item', formatter, className: 'local-tooltip', extraCssText: 'max-width:240px;' } } }));
  const tooltip = capture.props.option.tooltip;
  expect(tooltip.className).toBe('dashboard-chart-tooltip local-tooltip');
  expect(tooltip.position).toBeTypeOf('function');
  expect(tooltip.formatter).toBe(formatter);
  expect(tooltip.trigger).toBe('item');
  expect(tooltip.transitionDuration).toBe(0);
  expect(tooltip.extraCssText).toBe('max-width:240px;');
  renderToStaticMarkup(createElement(Chart, { option: { tooltip: { trigger: 'axis' } } }));
  expect(capture.props.option.tooltip.extraCssText).not.toContain('backdrop-filter');
});

test('shared tooltip positioning stays inside the chart and snaps to physical pixels under CSS zoom', () => {
  renderToStaticMarkup(createElement(Chart, { option: { tooltip: { trigger: 'axis' } } }));
  const variables = new Map();
  const zoom = 0.64755;
  const cssWidth = 1000.4;
  const origin = { left: 13.2, top: 27.8, width: cssWidth * zoom };
  const element = { offsetParent: { offsetWidth: 1000, getBoundingClientRect: () => origin }, style: { setProperty: (key, value) => variables.set(key, value) } };
  vi.stubGlobal('window', { devicePixelRatio: 1.875 });
  vi.stubGlobal('getComputedStyle', () => ({ width: `${cssWidth}px` }));
  try {
    for (const point of [[0, 0], [450, 80], [990, 270]]) {
      const position = capture.props.option.tooltip.position(point, [], element, null, { viewSize: [1000, 280], contentSize: [400, 120] });
      position.forEach((value, axis) => {
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThanOrEqual(axis === 0 ? 600 : 160);
        const pixel = ((axis === 0 ? origin.left : origin.top) + value * zoom) * 1.875;
        expect(pixel).toBeCloseTo(Math.round(pixel), 8);
      });
      expect(variables.get('--chart-tooltip-x')).toBe(`${position[0]}px`);
      expect(variables.get('--chart-tooltip-y')).toBe(`${position[1]}px`);
    }
  } finally {
    vi.unstubAllGlobals();
  }
});

test('shared charts keep deferred updates by default', () => {
  renderToStaticMarkup(createElement(Chart, { option: {} }));
  expect(capture.props.lazyUpdate).toBe(true);
});

test('interactive charts can render replacement models synchronously', () => {
  renderToStaticMarkup(createElement(Chart, { option: {}, lazyUpdate: false }));
  expect(capture.props.lazyUpdate).toBe(false);
});
