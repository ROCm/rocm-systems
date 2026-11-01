import { fontSizes, fontWeights } from '../../theme/typography';
import { Fragment } from 'react';
import { THREAD_POLICIES } from '../../data/runSchema';
import { Box, ButtonBase, Typography } from '@mui/material';
import { selectConfigurationComparison } from '../../data/branchSelectors';
import { formatDuration, formatPercent } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';
import { sansFont, shapeTokens } from '../../theme/tokens';
import CategoryTag from '../shared/CategoryTag';
import { configurationTargets } from './branchSelection';
import { configurationState } from './branchPresentation';

export default function ConfigurationMatrix({ candidate, reference, selection, suites, query = '', onSelect, runs = [candidate, reference] }) {
  const targets = configurationTargets(runs, selection.target);
  return <Box component="section" aria-labelledby="branch-matrix-heading" sx={{ mt: '20px', pt: 2, borderTop: 1, borderColor: 'divider' }}>
    <Box sx={{ display: 'flex', gap: 1, justifyContent: 'space-between', flexWrap: 'wrap', mb: '10px' }}>
      <Typography id="branch-matrix-heading" component="h3" variant="h3">Change by configuration</Typography>
      <Typography variant="caption" color="text.secondary">Matched benchmark wall time</Typography>
    </Box>
    <Box sx={{ display: 'grid', gridTemplateColumns: `max-content repeat(${Math.max(targets.length, 1)}, minmax(120px,1fr))`, minWidth: 0, overflowX: 'auto', gap: 1, alignItems: 'stretch' }}>
      <Box />
      {targets.map((target) => <Box key={target} sx={{ textAlign: 'center' }}><CategoryTag kind="target" label={target} /></Box>)}
      {THREAD_POLICIES.map((mode) => <Fragment key={mode}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}><CategoryTag kind="mode" label={mode} /></Box>
        {targets.map((target) => {
          const comparison = selectConfigurationComparison(candidate, reference, { target, mode, suites, query });
          const summary = configurationState(candidate, reference, comparison, target, mode);
          const selected = selection.target === target && selection.mode === mode;
          const tone = changeTone(classifyDurationChange(summary.available ? summary.deltaPercent : null));
          const direction = summary.deltaSeconds < 0 ? 'Less time' : summary.deltaSeconds > 0 ? 'More time' : 'Same measured time';
          const value = summary.available ? Number.isFinite(summary.deltaPercent) ? formatPercent(summary.deltaPercent) : `${summary.deltaSeconds > 0 ? '+' : summary.deltaSeconds < 0 ? '−' : ''}${formatDuration(Math.abs(summary.deltaSeconds))}` : 'Unavailable';
          return <ButtonBase key={target} data-testid={`branch-config-${target}-${mode}`} aria-pressed={selected}
            data-comparison-state={summary.available ? 'measured' : 'unavailable'}
            aria-label={`${target} ${mode} · ${summary.available ? `${value}, ${direction}, ${summary.matched} matched benchmarks` : summary.reason}`}
            aria-controls="branch-benchmark-differences" onClick={() => onSelect(target, mode)} sx={{
              minHeight: 88, padding: '14px 16px', display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', gap: '4px', textAlign: 'center', overflowWrap: 'anywhere',
              border: 1, borderColor: selected ? 'dashboard.selectionAccent' : 'dashboard.border',
              borderStyle: summary.available ? 'solid' : 'dashed', borderRadius: shapeTokens.cardRadius,
              bgcolor: summary.available && tone !== 'neutral' ? `dashboard.${tone}Surface` : selected ? 'action.selected' : 'background.paper',
              boxShadow: selected ? (theme) => `inset 0 0 0 1px ${theme.palette.dashboard.selectionAccent}` : 'none',
              '&:hover': { bgcolor: selected ? 'action.selected' : 'action.hover', borderColor: 'dashboard.controlBorder' },
            }}>
            <Typography component="span" sx={{ fontFamily: sansFont, fontSize: summary.available ? fontSizes.section : fontSizes.body, fontWeight: fontWeights.emphasized,
              color: !summary.available ? 'text.muted' : tone === 'neutral' ? 'text.secondary' : `${tone}.main` }}>{value}</Typography>
            <Typography component="span" variant="caption" color="text.secondary">{summary.available ? `${direction} · ${summary.matched} matched` : summary.reason}</Typography>
            {summary.available && summary.deltaPercent === null && <Typography component="span" variant="caption" color="text.secondary">{comparison.baselineDuration === 0 ? 'Percentage unavailable: zero reference runtime' : 'Percentage unavailable'}</Typography>}
          </ButtonBase>;
        })}
      </Fragment>)}
    </Box>

  </Box>;
}
