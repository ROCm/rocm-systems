import { fontSizes, fontWeights, textStyles } from '../../theme/typography';
import { useEffect, useState } from 'react';
import { Box, ButtonBase, Collapse, Tab, Tabs, Typography, useMediaQuery, useTheme } from '@mui/material';
import MenuBookRoundedIcon from '@mui/icons-material/MenuBookRounded';
import ShowChartRoundedIcon from '@mui/icons-material/ShowChartRounded';
import AccountTreeRoundedIcon from '@mui/icons-material/AccountTreeRounded';
import GridViewRoundedIcon from '@mui/icons-material/GridViewRounded';
import CompareArrowsRoundedIcon from '@mui/icons-material/CompareArrowsRounded';
import FiltersBar from './FiltersBar';
import SurfaceButton from '../shared/SurfaceButton';
import HoverTooltip from '../shared/HoverTooltip';
import { dashboardPages } from '../../config/dashboardPages';

const pageIcons = { overview: ShowChartRoundedIcon, branch: AccountTreeRoundedIcon, benchmarks: GridViewRoundedIcon, compare: CompareArrowsRoundedIcon };

const navigationTabStyles = {
  minWidth: 0, minHeight: 40, height: 40, maxWidth: 'none', padding: '10px 12px',
  flexDirection: 'row', justifyContent: 'flex-start', gap: '12px', borderRadius: '8px', textAlign: 'left',
  color: 'text.secondary', fontSize: fontSizes.nav, lineHeight: '24px', fontWeight: fontWeights.regular,
  '&:hover': { bgcolor: 'action.hover' },
  '&.Mui-selected': {
    bgcolor: 'action.selected', color: 'dashboard.sidebarAccentText', fontWeight: fontWeights.regular,
    boxShadow: (theme) => `inset 0 0 0 1px ${theme.palette.dashboard.selectionBorder}`,
    '&:hover': { bgcolor: 'action.selected' },
  },
  // Tab owns its icon slot's default margin, rather than the enclosing shell.
  '& > .MuiTab-icon': { m: 0 },
};

function UpdatedStatus({ generatedAt, loading }) {
  const timestamp = typeof generatedAt === 'string' ? Date.parse(generatedAt) : NaN;
  const available = !loading && Number.isFinite(timestamp);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!available) return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, [available]);

  const minutes = Math.floor(Math.max(0, now - timestamp) / 60_000);
  const age = minutes < 1 ? 'just now' : minutes < 60 ? `${minutes}m ago`
    : minutes < 1440 ? `${Math.floor(minutes / 60)}h ago`
      : minutes < 10080 ? `${Math.floor(minutes / 1440)}d ago` : `${Math.floor(minutes / 10080)}w ago`;

  return <Box data-testid="dashboard-updated" sx={{
    display: 'inline-flex', alignItems: 'center', alignSelf: 'flex-start', gap: '6px', mt: '10px', minWidth: '124px', height: '23px',
    padding: '2px 10px 2px 8px', border: 1, borderColor: 'dashboard.border', borderRadius: '999px', bgcolor: 'background.default',
    color: 'text.secondary', ...textStyles.captionCompact, whiteSpace: 'nowrap',
  }}>
    <Box component="span" aria-hidden="true" sx={{ width: 7, height: 7, flexShrink: 0, borderRadius: '50%', bgcolor: available ? 'dashboard.successGraphic' : 'text.muted' }} />
    {available ? <Box component="time" dateTime={generatedAt}>Updated {age}</Box> : loading ? 'Loading data…' : 'Data unavailable'}
  </Box>;
}

export default function DashboardShell({ data, state, loading = false, header, children }) {
  const theme = useTheme();
  const desktop = useMediaQuery(theme.breakpoints.up('md'));
  const [filtersOpen, setFiltersOpen] = useState(false);
  const targets = state.targets ?? [];
  const suites = state.suites ?? [];
  const modes = state.modes ?? [];
  const summary = `${targets.length} ${targets.length === 1 ? 'target' : 'targets'} / ${suites.length} ${suites.length === 1 ? 'suite' : 'suites'} / ${modes.length ? modes.join(' + ') : 'no thread policies'}`;

  return (
    <Box data-testid="dashboard-shell" sx={{ display: { xs: 'flex', md: 'grid' }, flexDirection: 'column', gridTemplateColumns: '256px minmax(0, 1fr)', minHeight: '100dvh' }}>
      <Box component="aside" aria-label="Dashboard navigation and filters" sx={{
        bgcolor: 'background.paper', borderRight: { md: `1px solid ${theme.palette.dashboard.border}` }, borderBottom: { xs: `1px solid ${theme.palette.dashboard.border}`, md: 0 },
        px: { xs: 2, md: '14px' }, pt: { md: '16px' }, minWidth: 0,
        position: { md: 'sticky' }, top: { md: 0 },
        height: { md: '100dvh' },
        overflowY: { md: 'auto' }, overflowX: { md: 'hidden' },
        scrollbarGutter: { md: 'stable' }, alignSelf: { xs: 'stretch', md: 'start' },
        '--sidebar-navigation-gap': '6px',
        '--sidebar-group-top': '18px', '--sidebar-group-bottom': '14px',
        '--sidebar-chip-row-gap': '11px',
      }}>
        <Box sx={{ px: 0, pt: '4px', pb: '16px', display: 'flex', flexDirection: 'column', justifyContent: 'center', mb: '16px', '& > *': { ml: '6px' }, borderBottom: `1px solid ${theme.palette.dashboard.border}` }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', height: textStyles.brand.lineHeight }}>
            <Typography data-testid="dashboard-wordmark"
              sx={{ ...textStyles.brand, letterSpacing: '-1px', color: 'dashboard.sidebarAccentText' }}>rocjitsu</Typography>
            <HoverTooltip title="rocjitsu user guide"><ButtonBase component="a" href="https://rocm.github.io/rocm-systems/rocjitsu/" target="_blank" rel="noopener noreferrer"
              aria-label="rocjitsu user guide (opens in a new tab)" sx={{
                display: 'grid', placeItems: 'center', width: '30px', height: '30px', flexShrink: 0,
                border: 1, borderColor: 'dashboard.border', borderRadius: '8px', bgcolor: 'background.paper', color: 'dashboard.sidebarAccentText',
                '&:hover': { bgcolor: 'action.selected', borderColor: 'dashboard.selectionBorder' },
              }}><MenuBookRoundedIcon sx={{ width: 16, height: 16, fontSize: 16 }} /></ButtonBase></HoverTooltip>
          </Box>
          {/* 12px so the uppercase subtitle stays on one line in the 256px sidebar; at the 14px caption size it wraps. */}
          <Typography sx={{ mt: '2px', ml: '6px', ...textStyles.captionCompact, fontSize: '12px', color: 'text.secondary', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Simulator perf dashboard</Typography>
          <UpdatedStatus key={data.generatedAt ?? 'missing'} generatedAt={data.generatedAt} loading={loading} />
        </Box>
        <Box component="nav" aria-label="Dashboard">
          <Tabs data-testid="dashboard-navigation" value={state.tab} orientation={desktop ? 'vertical' : 'horizontal'}
            onChange={(_, value) => state.setTab(value)} aria-label="Dashboard views" sx={{ minHeight: 40, mb: '14px' }} slotProps={{
              list: { sx: { display: { xs: 'grid', md: 'flex' }, gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2px', rowGap: 'var(--sidebar-navigation-gap, 2px)' } },
              indicator: { sx: { display: 'none' } },
            }}>
            {dashboardPages.map(({ id, label }) => {
              const Icon = pageIcons[id];
              return <Tab key={id} value={id} sx={navigationTabStyles} icon={Icon ? <Icon sx={{ fontSize: 20, width: 20, height: 20, color: state.tab === id ? 'dashboard.selectionAccent' : 'inherit' }} /> : undefined} iconPosition="start" label={label} id={`dashboard-tab-${id}`} aria-controls={`dashboard-panel-${id}`} />;
            })}
          </Tabs>
        </Box>
        {state.tab === 'branch' ? (
          <Box sx={{ borderTop: `1px solid ${theme.palette.dashboard.border}`, px: 1, py: '20px' }}>
            <Typography variant="caption" color="text.secondary">Branch runs uses its local configuration matrix. Canonical filters are not applied here.</Typography>
          </Box>
        ) : <>
          <SurfaceButton aria-expanded={filtersOpen} aria-controls="dashboard-filters" onClick={() => setFiltersOpen((open) => !open)}
            sx={{ display: { xs: 'flex', md: 'none' }, my: '12px', mx: 1, minHeight: 44, maxWidth: 'calc(100% - 16px)', textAlign: 'left', justifyContent: 'flex-start', borderColor: 'dashboard.controlBorder', fontSize: fontSizes.body, lineHeight: '20px', px: '12px' }}>Filters · {summary}</SurfaceButton>
          <Box id="dashboard-filters">
            <Collapse in={desktop || filtersOpen} unmountOnExit>
              <FiltersBar data={data} state={state} disabled={loading || !(state.tab === 'compare' ? data.allRuns : data.runs)?.length} />
            </Collapse>
          </Box>
        </>}
      </Box>
      <Box sx={{ minWidth: 0, flex: 1, display: 'flex', flexDirection: 'column', '& > main': { flex: 1 } }}>
        {header}
        {import.meta.env.MODE === 'fixtures' && <Box role="note" data-testid="fixture-data-warning" sx={{ borderBottom: 1, borderColor: 'dashboard.border', bgcolor: 'dashboard.warningSurface', color: 'dashboard.warningText', px: { xs: 2, md: 3 }, py: 1, fontSize: fontSizes.caption, lineHeight: '18px' }}>
          <Box component="strong" sx={{ fontWeight: fontWeights.emphasized }}>Test fixtures — fictional measurements</Box>{' · Not production data.'}
        </Box>}
        {children}
      </Box>
    </Box>
  );
}
