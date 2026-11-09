import { useCallback, useLayoutEffect, useRef } from 'react';
import { bindChartNativeInput } from './chartNativeInput';
import { positionChartTooltip } from './chartTooltip';
import ReactEChartsCore from 'echarts-for-react/esm/core';
import * as echarts from 'echarts/core';
import { BarChart, LineChart, ScatterChart } from 'echarts/charts';
import {
  GridComponent,
  MarkLineComponent,
  TooltipComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import { useTheme } from '@mui/material/styles';

echarts.use([
  BarChart,
  LineChart,
  ScatterChart,
  GridComponent,
  MarkLineComponent,
  TooltipComponent,
  CanvasRenderer,
]);

export default function Chart({ option, height = 280, ariaLabel, ariaDescribedBy, onEvents, onReady, lazyUpdate = true }) {
  const cleanupInput = useRef(null);
  useLayoutEffect(() => () => cleanupInput.current?.(), []);
  const handleReady = useCallback((chart) => {
    cleanupInput.current?.();
    cleanupInput.current = bindChartNativeInput(chart);
    onReady?.(chart);
  }, [onReady]);
  const theme = useTheme();
  const tooltip = option.tooltip ? {
    backgroundColor: theme.palette.background.paper,
    borderColor: theme.palette.divider,
    borderWidth: 1,
    borderRadius: 10,
    padding: [10, 12],
    confine: true,
    transitionDuration: 0,
    extraCssText: `box-shadow: ${theme.shadows[5]};`,
    position: positionChartTooltip,
    ...option.tooltip,
    className: ['dashboard-chart-tooltip', option.tooltip.className].filter(Boolean).join(' '),
    textStyle: {
      color: theme.palette.text.primary,
      fontFamily: theme.typography.fontFamily,
      ...option.tooltip.textStyle,
    },
  } : undefined;
  const themedOption = {
    animationDuration: 650,
    animationDurationUpdate: 350,
    animationEasing: 'cubicOut',
    animationEasingUpdate: 'cubicInOut',
    textStyle: { color: theme.palette.text.secondary, fontFamily: theme.typography.fontFamily },
    ...option,
    ...(tooltip && { tooltip }),
  };

  return (
    <ReactEChartsCore
      echarts={echarts}
      option={themedOption}
      notMerge
      lazyUpdate={lazyUpdate}
      onEvents={onEvents}
      onChartReady={handleReady}
      opts={{ renderer: 'canvas' }}
      style={{ width: '100%', height }}
      aria-label={ariaLabel}
      aria-describedby={ariaDescribedBy}
      role="img"
    />
  );
}
