import { useState } from 'react';
import { alpha, Box, Button, Collapse, Tab, Tabs, Typography, useMediaQuery, useTheme } from '@mui/material';
import ShowChartRoundedIcon from '@mui/icons-material/ShowChartRounded';
import AccountTreeRoundedIcon from '@mui/icons-material/AccountTreeRounded';
import GridViewRoundedIcon from '@mui/icons-material/GridViewRounded';
import CompareArrowsRoundedIcon from '@mui/icons-material/CompareArrowsRounded';
import FiltersBar from './FiltersBar';

const pages = [
  ['overview', 'Overview', ShowChartRoundedIcon], ['branch', 'Branch Runs', AccountTreeRoundedIcon],
  ['benchmarks', 'Benchmarks', GridViewRoundedIcon], ['compare', 'Run Comparison', CompareArrowsRoundedIcon],
];

export default function DashboardShell({ data, state, loading = false, header, children }) {
  const theme = useTheme();
  const desktop = useMediaQuery(theme.breakpoints.up('md'));
  const [filtersOpen, setFiltersOpen] = useState(false);
  const targets = state.targets ?? [];
  const suites = state.suites ?? [];
  const modes = state.modes ?? [];
  const summary = `${targets.length} ${targets.length === 1 ? 'target' : 'targets'} / ${suites.length} ${suites.length === 1 ? 'suite' : 'suites'} / ${modes.length ? modes.join(' + ') : 'no modes'}`;

  return (
    <Box data-testid="dashboard-shell" sx={{ display: { xs: 'block', md: 'grid' }, gridTemplateColumns: '228px minmax(0, 1fr)', minHeight: '100vh' }}>
      <Box component="aside" aria-label="Dashboard navigation and filters" sx={{
        bgcolor: 'action.hover', borderRight: { md: `2px solid ${theme.palette.divider}` }, borderBottom: { xs: `1px solid ${theme.palette.divider}`, md: 0 },
        px: { xs: 2, md: '14px' }, alignSelf: 'stretch', minWidth: 0,
      }}>
        <Box sx={{ px: 0.75, py: { xs: 1.5, md: 2.5 }, minHeight: { xs: 82, md: 110 }, display: 'flex', flexDirection: 'column', justifyContent: 'center', mb: { xs: 1.25, md: 2.25 }, borderBottom: `1px solid ${alpha(theme.palette.text.primary, 0.3)}` }}>
          <Typography data-testid="dashboard-wordmark" sx={{ fontSize: { xs: 28, md: 32 }, lineHeight: { xs: '36px', md: '40px' }, fontWeight: 600, letterSpacing: '-1px' }}>Rocjitsu</Typography>
          <Typography sx={{ fontSize: 10, lineHeight: '16px', color: 'text.secondary', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Sim perf dashboard</Typography>
        </Box>
        <Box component="nav" aria-label="Dashboard">
          <Typography variant="overline" sx={{ display: { xs: 'none', md: 'block' }, px: 1, mb: 1, color: 'text.secondary' }}>Dashboard</Typography>
          <Tabs data-testid="dashboard-navigation" value={state.tab} orientation={desktop ? 'vertical' : 'horizontal'}
            onChange={(_, value) => state.setTab(value)} aria-label="Dashboard views" sx={{
              minHeight: 44, mb: { xs: 1.25, md: 3 },
              '& .MuiTabs-list': { display: { xs: 'grid', md: 'flex' }, gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '5px' },
              '& .MuiTabs-indicator': { display: 'none' },
              '& .MuiTab-root': {
                minWidth: 0, minHeight: { xs: 44, md: 42 }, maxWidth: 'none', px: { xs: 1.25, md: 1.5 }, py: 1.25,
                flexDirection: 'row', justifyContent: 'flex-start', gap: 1.25, borderRadius: '7px', textAlign: 'left',
                color: 'text.secondary', fontSize: { xs: 13, md: 14 }, lineHeight: '20px', fontWeight: 500,
              },
              '& .MuiTab-root.Mui-selected': { bgcolor: 'action.selected', color: 'selectionText', fontWeight: 600, boxShadow: `inset 0 0 0 1px ${theme.palette.dashboard.selectionBorder}` },
              '& .MuiTab-icon': { m: 0, fontSize: 18, opacity: 0.85 },
              '& .MuiTab-root.Mui-selected::after': { content: '""', display: { xs: 'none', md: 'block' }, ml: 'auto', width: 5, height: 5, flexShrink: 0, borderRadius: '50%', bgcolor: 'primary.main' },
            }}>
            {pages.map(([value, label, Icon]) => <Tab key={value} value={value} icon={<Icon />} iconPosition="start" label={label} id={`dashboard-tab-${value}`} aria-controls={`dashboard-panel-${value}`} />)}
          </Tabs>
        </Box>
        {state.tab === 'branch' ? (
          <Box sx={{ borderTop: `1px solid ${alpha(theme.palette.text.primary, 0.3)}`, px: 1, py: 2.5 }}>
            <Typography variant="caption" color="text.secondary">Branch Runs uses its local configuration matrix. Canonical filters are not applied here.</Typography>
          </Box>
        ) : <>
          <Button variant="outlined" color="inherit" aria-expanded={filtersOpen} aria-controls="dashboard-filters" onClick={() => setFiltersOpen((open) => !open)}
            sx={{ display: { xs: 'flex', md: 'none' }, my: 1.5, mx: 1, minHeight: 44, maxWidth: 'calc(100% - 16px)', textAlign: 'left', justifyContent: 'flex-start', borderColor: 'dashboard.controlBorder', color: 'text.secondary', fontSize: 14, lineHeight: '20px', px: 1.5 }}>Filters · {summary}</Button>
          <Box id="dashboard-filters">
            <Collapse in={desktop || filtersOpen} unmountOnExit>
              <FiltersBar data={data} state={state} disabled={loading || !(state.tab === 'compare' ? data.allRuns : data.runs)?.length} />
            </Collapse>
          </Box>
        </>}
      </Box>
      <Box sx={{ minWidth: 0, display: 'flex', flexDirection: 'column', '& > main': { flex: 1 } }}>
        {header}
        {import.meta.env.MODE === 'fixtures' && <Box role="note" data-testid="fixture-data-warning" sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'dashboard.warningSurface', color: 'dashboard.warningText', px: { xs: 2, md: 3 }, py: 1, fontSize: 12, lineHeight: '18px' }}>
          <Box component="strong" sx={{ fontWeight: 600 }}>Test fixtures — fictional measurements</Box>{' · Not production data.'}
        </Box>}
        {children}
      </Box>
    </Box>
  );
}
