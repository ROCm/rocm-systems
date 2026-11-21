import { fontSizes } from '../../theme/typography';
import AnchoredSelect from '../shared/AnchoredSelect';
import useElementWidth from '../../hooks/useElementWidth';
import { memo, useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { Box, MenuItem, Stack, ToggleButton, ToggleButtonGroup, Typography, useTheme } from '@mui/material';
import Chart from '../shared/Chart';
import SectionCard from '../shared/SectionCard';
import CatalogWorkloads from './CatalogWorkloads';
import { escapeHtml, formatDuration, formatFullDate, shortSha } from '../../utils/formatters';
import { targetColor } from '../../theme/targetColors';
import { CHART_GRID_INSETS, chartLineStyle, chartPointStyle, chartPointSize, chartSwatch } from '../../utils/chartStyles';
import ThreadPolicySwatch from '../shared/ThreadPolicySwatch';
import { alpha } from '@mui/material/styles';
import { historyRanges } from '../../config/historyRanges';
import { CONTROL_HEIGHT, controlGroupSx } from '../shared/segmentedControlSx';
import { dayAxisBounds, dayXAxis } from '../../utils/dayAxis';
import { visuallyHiddenStyles } from '../../theme/styles';
import { commitTimestampFor } from '../../data/runOrdering';
import { trendEmptyMessage, trendKeyIndex, nearestTrendIndex, trendSlotStatus } from './overviewPresentation';

const StableChart = memo(Chart);

const controlSelectSx = {
  width: 'fit-content', maxWidth: '100%', height: CONTROL_HEIGHT, fontSize: fontSizes.body,
  // No floating label: drop MUI's empty notch legend, whose theme padding would cut a gap in the top border.
  // MUI also shifts the fieldset up 5px for the legend; reset it so the outline matches the control height.
  '& .MuiOutlinedInput-notchedOutline': { top: 0 },
  '& .MuiOutlinedInput-notchedOutline legend': { display: 'none' },
  '& .MuiSelect-select': { display: 'flex', alignItems: 'center', boxSizing: 'border-box', height: `${CONTROL_HEIGHT}px !important`, minHeight: 0, py: 0, whiteSpace: 'nowrap' },
};



export default function DurationHistory({ projection, catalogs, onRangeChange, onCatalogChange }) {
  const theme = useTheme();
  const helpId = useId();
  const catalogLabelId = useId();
  const chartRef = useRef(null);
  const [widthRef, chartWidth] = useElementWidth();
  const guideRef = useRef(null);
  const focusRef = useRef(null);
  const pointerSource = useRef('mouse');
  const [inspection, setInspection] = useState(null);
  const inspected = inspection?.projection === projection ? inspection.index : null;
  const eligible = projection.slots.flatMap((slot, index) => slot.run && projection.rows.some((r) => Number.isFinite(r.points[index]?.speedup)) ? [index] : []);
  const hasCandidates = (projection.attempts?.length ?? projection.slots.filter((slot) => slot.run).length) > 0;
  const keyboardIndexes = hasCandidates && projection.rows.some((row) => row.cohortIds.length)
    ? projection.slots.map((_, index) => index) : [];
  const statusRef = useRef(null);
  const announcedProjection = useRef(projection);
  // One persistent live-region node owns both keyboard feedback and scope notices.
  // Like the transient chart overlays, its DOM is updated without prop-sync state.
  const announce = useCallback((text, visible = false) => {
    const node = statusRef.current;
    if (!node) return;
    node.textContent = text;
    node.setAttribute('data-visible', String(visible));
    if (text) node.setAttribute('role', 'status');
    else node.removeAttribute('role');
  }, []);
  const colors = projection.rows.map((row) => targetColor(theme, row.target));
  const clear = () => {
    const chart = chartRef.current;
    if (chart && !chart.isDisposed()) chart.dispatchAction({ type: 'hideTip' });
    if (guideRef.current) guideRef.current.style.display = 'none';
    if (focusRef.current) focusRef.current.style.display = 'none';
  };
  useEffect(() => {
    // Projection changes invalidate imperative hover DOM as well as logical inspection.
    if (guideRef.current) guideRef.current.style.display = 'none';
    if (focusRef.current) { focusRef.current.style.display = 'none'; focusRef.current.innerHTML = ''; }
    const chart = chartRef.current;
    if (chart && !chart.isDisposed()) chart.dispatchAction({ type: 'hideTip' });
    if (announcedProjection.current !== projection) {
      announce('Comparison scope updated. Use arrow keys to inspect dates and attempts.');
      announcedProjection.current = projection;
    }
  }, [projection, announce]);
  const inspect = (index) => {
    if (index == null) { clear(); return; }
    const source = pointerSource.current;
    setInspection((current) => current?.projection === projection && current.index === index && current.source === source
      ? current : { projection, index, source });
    announce(trendSlotStatus(projection, index), source === 'keyboard');
    const chart = chartRef.current;
    const hits = projection.rows.flatMap((r, si) => Number.isFinite(r.points[index]?.speedup) ? [si] : []);
    if (!hits.length || !chart || chart.isDisposed()) { clear(); return; }
    const point = chart.convertToPixel({ seriesIndex: hits[0] }, [projection.slots[index].x, projection.rows[hits[0]].points[index].speedup]);
    const grid = chart.getModel().getComponent('grid').coordinateSystem.getRect();
    // Series lookup can put an endpoint a floating-point fraction outside the
    // grid (e.g. 1059.0000000000002 > 1059), suppressing its axis tooltip.
    chart.dispatchAction({ type: 'showTip',
      x: Math.max(grid.x + 1e-6, Math.min(grid.x + grid.width - 1e-6, point[0])),
      y: Math.max(grid.y + 1e-6, Math.min(grid.y + grid.height - 1e-6, point[1])),
    });
    const x = chart.convertToPixel({ xAxisIndex: 0 }, projection.slots[index].x);
    Object.assign(guideRef.current.style, { display: 'block', left: `${x}px` });
    // Temporary emphasis is a small segment overlay, not persistent point dots.
    focusRef.current.innerHTML = hits.map((si) => {
      const row = projection.rows[si];
      const center = chart.convertToPixel({ seriesIndex: si }, [projection.slots[index].x, row.points[index].speedup]);
      const paths = [index - 1, index + 1].filter((i) => Number.isFinite(row.points[i]?.speedup)).map((i) => {
        const other = chart.convertToPixel({ seriesIndex: si }, [projection.slots[i].x, row.points[i].speedup]);
        const fraction = Math.min(1, 8 / Math.max(1, Math.abs(other[0] - center[0])));
        return `M${center[0]},${center[1]} L${center[0] + (other[0] - center[0]) * fraction},${center[1] + (other[1] - center[1]) * fraction}`;
      }).join(' ');
      return `<path d="${paths}" fill="none" stroke="${colors[si]}" stroke-width="3"/><circle cx="${center[0]}" cy="${center[1]}" r="7" fill="${theme.palette.background.paper}" stroke="${colors[si]}" stroke-width="3"/>`;
    }).join('');
    focusRef.current.style.display = 'block';
  };
  const firstEligible = eligible[0];
  const bounds = useMemo(() => dayAxisBounds(projection, firstEligible), [projection, firstEligible]);

  const option = useMemo(() => ({
    animation: false,
    grid: { ...CHART_GRID_INSETS, top: 26, bottom: 42 },
    tooltip: { trigger: 'axis', triggerOn: 'none', axisPointer: { type: 'none' }, hideDelay: 0,
      className: 'performance-trend-tooltip',
      formatter: (parameters) => {
        const index = (Array.isArray(parameters) ? parameters[0] : parameters)?.dataIndex;
        const slot = projection.slots[index];
        const valid = projection.rows.filter((row) => Number.isFinite(row.points[index]?.speedup));
        if (!slot?.run || !valid.length) return '';
        return `<div><b>Commit ${escapeHtml(shortSha(slot.run))}</b></div><div>${escapeHtml(formatFullDate(commitTimestampFor(slot.run)))}</div>` + valid.map((row) => {
          const point = row.points[index];
          return `<div class="performance-trend-tooltip-row">${chartSwatch(targetColor(theme, row.target), row.mode)}${escapeHtml(row.target)} ${escapeHtml(row.mode)} · <b>${point.speedup.toFixed(3)}×</b> · ${formatDuration(point.durationSeconds)} · ref ${formatDuration(row.reference.durationSeconds)}</div>`;
        }).join('');
      } },
    xAxis: dayXAxis(bounds, theme),
    yAxis: { type: 'value', scale: true, min: (extent) => extent.min < 1 ? extent.min - 0.01 : 1, max: (extent) => Math.max(1.01, extent.max + 0.01),
      nameTextStyle: { align: 'left', color: theme.palette.text.secondary }, axisLine: { show: true, lineStyle: { width: 1, color: theme.palette.dashboard.border } }, axisLabel: { formatter: (v) => `${v.toFixed(2)}×`, color: theme.palette.text.secondary, fontSize: fontSizes.caption }, splitLine: { lineStyle: { color: alpha(theme.palette.dashboard.border, 0.55), width: 1 } } },
    series: projection.rows.map((row, index) => ({ name: `${row.target} ${row.mode}`, type: 'line',
      data: row.points.map((point, i) => [projection.slots[i].x, point.speedup]), connectNulls: true,
      showSymbol: true, symbol: row.mode === 'single' ? 'diamond' : 'circle', symbolSize: (_value, params) => chartPointSize(row.points, params.dataIndex, row.mode, 'speedup', chartWidth),
      itemStyle: chartPointStyle(targetColor(theme, row.target), theme.palette.background.paper, row.mode),
      emphasis: { disabled: true }, lineStyle: chartLineStyle(targetColor(theme, row.target), row.mode),
      ...(index === 0 ? { markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { width: 1.4, opacity: 0.9, type: [14, 5, 3, 5], color: theme.palette.text.secondary }, data: [{ yAxis: 1 }] } } : {}) })),
  }), [projection, theme, bounds, chartWidth]);
  const onReady = useMemo(() => (chart) => { chartRef.current = chart; }, []);
  const pointer = (event) => {
    const chart = chartRef.current;
    if (!chart || chart.isDisposed()) return;
    const rect = chart.getDom().getBoundingClientRect();
    const pixel = [(event.clientX - rect.left) * chart.getWidth() / rect.width, (event.clientY - rect.top) * chart.getHeight() / rect.height];
    // A point on the plot edge can round just outside it.
    // Accept one viewport pixel of rounding, then clamp back into the grid.
    const grid = chart.getModel().getComponent('grid').coordinateSystem.getRect();
    const tolerance = Math.max(chart.getWidth() / rect.width, chart.getHeight() / rect.height);
    if (pixel[0] < grid.x - tolerance || pixel[0] > grid.x + grid.width + tolerance
      || pixel[1] < grid.y - tolerance || pixel[1] > grid.y + grid.height + tolerance) { clear(); return; }
    pixel[0] = Math.max(grid.x, Math.min(grid.x + grid.width, pixel[0]));
    pixel[1] = Math.max(grid.y, Math.min(grid.y + grid.height, pixel[1]));
    inspect(nearestTrendIndex(projection, chart.convertFromPixel('grid', pixel)[0]));
  };
  return <SectionCard data-testid="performance-trend" title="Performance trend" subtitle="Relative speedup · higher is faster · develop only" sx={{ height: '100%', minWidth: 0, display: 'flex', flexDirection: 'column' }} contentSx={{ flex: 1, display: 'flex', flexDirection: 'column', '& > *': { flexShrink: 0 }, '&:last-child': { pb: '8px' } }} action={<Box sx={{ maxWidth: '100%', overflowX: 'auto', p: '3px', m: '-3px', alignSelf: { sm: 'center' } }}><ToggleButtonGroup exclusive size="small" value={projection.range} onChange={(_, next) => next && onRangeChange(next)} aria-label="History timeframe" sx={controlGroupSx}>{historyRanges.map(({ id, label }) => <ToggleButton key={id} value={id} aria-label={label}>{id}</ToggleButton>)}</ToggleButtonGroup></Box>}>
    <Stack direction="row" sx={{ alignItems: 'center', flexWrap: 'wrap', gap: 1, mb: 1 }}>
      <Typography id={catalogLabelId} variant="caption" color="text.secondary">Benchmark set</Typography>
      <AnchoredSelect labelId={catalogLabelId} value={projection.catalogId ?? ''} onChange={(e) => { clear(); onCatalogChange(e.target.value || null); }} size="small" displayEmpty MenuProps={{ slotProps: { paper: { sx: { maxWidth: 'calc(100vw - 32px)' } } } }} sx={controlSelectSx}>
        <MenuItem value="" sx={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>Shared benchmarks</MenuItem>{projection.catalogId && !catalogs.some(({ catalog }) => catalog.id === projection.catalogId) && <MenuItem value={projection.catalogId} sx={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>Unavailable benchmark set: {projection.catalogId}</MenuItem>}
        {catalogs.map(({ catalog, label }) => <MenuItem key={catalog.id} value={catalog.id} sx={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{label}</MenuItem>)}
      </AnchoredSelect>
      <Typography variant="caption" color="text.secondary">1.00× = each configuration’s first valid positive point in range.</Typography>
    </Stack>
    <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 2, mb: 1, color: 'text.secondary' }}>{projection.rows.map((row, index) => <Typography key={row.key} variant="caption" sx={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}><ThreadPolicySwatch policy={row.mode} color={colors[index]} legend />{row.target} {row.mode}</Typography>)}</Stack>
    <Box ref={widthRef} data-testid="performance-trend-chart" tabIndex={keyboardIndexes.length ? 0 : undefined} role="group" aria-label="Inspect performance trend" aria-describedby={helpId}
      onKeyDown={(e) => {
        if (!keyboardIndexes.length || !['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Escape'].includes(e.key)) return;
        e.preventDefault();
        pointerSource.current = 'keyboard';
        inspect(trendKeyIndex(keyboardIndexes, inspected, e.key));
        if (e.key === 'Escape') { setInspection(null); announce(''); }
      }}
      onPointerMove={(e) => { if (e.pointerType === 'mouse' || e.pointerType === 'pen') { pointerSource.current = e.pointerType; pointer(e); } }}
      onPointerDown={(e) => { pointerSource.current = e.pointerType; pointer(e); }}
      onPointerLeave={() => { if (['mouse', 'pen'].includes(pointerSource.current)) clear(); }}
      sx={{ position: 'relative', mt: 1, flex: '0 0 300px', height: 300 }}>
      {!eligible.length ? <Box key="empty-comparison" data-testid="speedup-empty-state" sx={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', textAlign: 'center', px: 2 }}><Typography variant="body2" color="text.secondary">{trendEmptyMessage(projection)}</Typography></Box> : <>
        <Typography variant="caption" sx={{ position: 'absolute', top: 0, left: 0, color: 'text.secondary', pointerEvents: 'none', zIndex: 1 }}>Relative speedup (×)</Typography>
        <Box component="svg" ref={guideRef} data-testid="point-hover-guide" aria-hidden="true" sx={{ display: 'none', pointerEvents: 'none', position: 'absolute', top: 26, width: '1px', height: 'calc(100% - 68px)', color: 'text.secondary', overflow: 'visible', zIndex: 1 }}><line x1="0" x2="0" y1="0" y2="100%" stroke="currentColor" strokeWidth="1" strokeDasharray="2 3" /></Box>
        <Box component="svg" ref={focusRef} data-testid="hover-line-focus" aria-hidden="true" sx={{ position: 'absolute', inset: 0, width: '100%', height: '100%', display: 'none', pointerEvents: 'none', zIndex: 2 }} />
        <Box sx={{ position: 'absolute', inset: 0 }}><StableChart option={option} height="100%" ariaLabel="Relative speedup by configuration" lazyUpdate={false} onReady={onReady} /></Box>
      </>}
    </Box>
    <Typography ref={statusRef} data-testid="trend-inspection-status" aria-live="polite" aria-atomic="true" data-visible="false" variant="caption" sx={{ display: 'block', mt: 1, overflowWrap: 'anywhere', '&[data-visible="false"]': visuallyHiddenStyles }} />
    <Typography id={helpId} variant="caption" color="text.muted" sx={{ fontSize: fontSizes.caption }}>Arrow keys inspect dates and attempts · Home/End first/last · Escape clears · Date</Typography>
    <CatalogWorkloads catalogs={catalogs} />

  </SectionCard>;
}
