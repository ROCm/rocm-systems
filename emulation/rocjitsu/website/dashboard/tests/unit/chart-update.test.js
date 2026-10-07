import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import Chart from '../../src/components/shared/Chart.jsx';

const capture = vi.hoisted(() => ({ props: null }));
vi.mock('echarts-for-react/esm/core', () => ({ default: (props) => { capture.props = props; return createElement('div'); } }));

test('shared charts keep deferred updates by default', () => {
  renderToStaticMarkup(createElement(Chart, { option: {} }));
  expect(capture.props.lazyUpdate).toBe(true);
});

test('interactive charts can render replacement models synchronously', () => {
  renderToStaticMarkup(createElement(Chart, { option: {}, lazyUpdate: false }));
  expect(capture.props.lazyUpdate).toBe(false);
});
