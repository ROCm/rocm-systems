import { Fragment } from 'react';
import { Box, ButtonBase, Typography } from '@mui/material';
import { selectConfigurationComparison } from '../../data/branchSelectors';
import { formatDuration, formatPercent } from '../../utils/formatters';
import { monoFont } from '../../theme/tokens';
import CategoryTag from '../shared/CategoryTag';
import { configurationState } from './branchPresentation';

export default function ConfigurationMatrix({ candidate, reference, selection, suites, query = '', onSelect }) {
  return <Box component="section" aria-labelledby="branch-matrix-heading" sx={{ mt: 2.5 }}>
    <Box sx={{ display: 'flex', gap: 1, justifyContent: 'space-between', flexWrap: 'wrap', mb: 1.5 }}>
      <Typography id="branch-matrix-heading" component="h3" variant="subtitle1" sx={{ fontWeight: 600 }}>Change by configuration</Typography>
      <Typography variant="caption" color="text.secondary">Matched benchmark wall time</Typography>
    </Box>
    <Box sx={{ display: 'grid', gridTemplateColumns: '40px minmax(0,1fr) minmax(0,1fr)', gap: 1, alignItems: 'stretch' }}>
      <Box />
      {['gfx1250', 'gfx950'].map((target) => <Box key={target} sx={{ textAlign: 'center' }}><CategoryTag kind="target" label={target} /></Box>)}
      {['ST', 'MT'].map((mode) => <Fragment key={mode}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}><CategoryTag kind="mode" label={mode} /></Box>
        {['gfx1250', 'gfx950'].map((target) => {
          const comparison = selectConfigurationComparison(candidate, reference, { target, mode, suites, query });
          const summary = configurationState(candidate, reference, comparison, target, mode);
          const selected = selection.target === target && selection.mode === mode;
          const direction = summary.deltaSeconds < 0 ? 'Less time' : summary.deltaSeconds > 0 ? 'More time' : 'Same measured time';
          const value = summary.available ? Number.isFinite(summary.deltaPercent) ? formatPercent(summary.deltaPercent) : `${summary.deltaSeconds > 0 ? '+' : summary.deltaSeconds < 0 ? '−' : ''}${formatDuration(Math.abs(summary.deltaSeconds))}` : 'Unavailable';
          return <ButtonBase key={target} data-testid={`branch-config-${target}-${mode}`} aria-pressed={selected}
            data-comparison-state={summary.available ? 'measured' : 'unavailable'}
            aria-label={`${target} ${mode === 'ST' ? 'single-threaded (ST)' : 'multi-threaded (MT)'} · ${summary.available ? `${value}, ${direction}, ${summary.matched} matched benchmarks` : summary.reason}`}
            aria-controls="branch-benchmark-differences" onClick={() => onSelect(target, mode)} sx={{
              minHeight: 88, p: { xs: 1, sm: 1.5 }, display: 'flex', flexDirection: 'column', alignItems: 'flex-start', justifyContent: 'center', gap: 0.5,
              border: 1, borderColor: selected ? 'primary.main' : 'divider', borderRadius: 1,
              bgcolor: selected ? 'action.selected' : 'transparent', textAlign: 'left', overflowWrap: 'anywhere',
              '&:hover': { bgcolor: 'action.hover' }, '&.Mui-focusVisible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: 2 },
            }}>
            <Typography component="span" sx={{ fontFamily: monoFont, fontSize: summary.available ? 20 : 14, fontWeight: 600,
              color: summary.available && Number.isFinite(summary.deltaPercent) && summary.deltaPercent < -3 ? 'success.main' : summary.available && summary.deltaPercent > 3 ? 'error.main' : 'text.secondary' }}>{value}</Typography>
            <Typography component="span" variant="caption" color="text.secondary">{summary.available ? `${direction} · ${summary.matched} matched` : summary.reason}</Typography>
            {summary.available && summary.deltaPercent === null && <Typography component="span" variant="caption" color="text.secondary">Percentage unavailable: zero reference runtime</Typography>}
          </ButtonBase>;
        })}
      </Fragment>)}
    </Box>
    <Typography variant="caption" component="p" color="text.secondary" sx={{ mt: 1 }}>ST: single-threaded · MT: multi-threaded. Select any cell to inspect its results, including unavailable configurations.</Typography>
  </Box>;
}
