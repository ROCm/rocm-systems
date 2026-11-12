import { fontSizes, fontWeights, sansFont } from '../../theme/typography';
import { Fragment } from 'react';
import { THREAD_POLICIES } from '../../data/runSchema';
import { Box, ButtonBase, Typography } from '@mui/material';
import { selectConfigurationComparison } from '../../data/branchSelectors';
import { formatDuration, formatPercent } from '../../utils/formatters';
import { changeTone, classifyDurationChange } from '../../utils/performance';
import { shapeTokens } from '../../theme/tokens';
import CategoryTag from '../shared/CategoryTag';
import { configurationTargets } from './branchSelection';
import { configurationState } from './branchPresentation';

// A change this large (or larger) fills the whole bar.
const MAGNITUDE_FULL_SCALE_PERCENT = 20;

export default function ConfigurationMatrix({ candidate, reference, selection, suites, query = '', onSelect, runs = [candidate, reference] }) {
  const targets = configurationTargets(runs, selection.target);
  return <Box component="section" aria-labelledby="branch-matrix-heading" sx={{ mt: '20px', pt: 2, borderTop: 1, borderColor: 'divider' }}>
    <Box sx={{ display: 'flex', gap: 1, justifyContent: 'space-between', flexWrap: 'wrap', mb: '10px' }}>
      <Typography id="branch-matrix-heading" component="h3" variant="h2">Change by configuration</Typography>
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
              border: 1, borderColor: 'dashboard.border', borderStyle: summary.available ? 'solid' : 'dashed', borderRadius: shapeTokens.cardRadius,
              // Cells stay neutral; tone lives in the arrow, number and magnitude bar. Selection: tinted fill + 2px ring.
              position: 'relative', overflow: 'hidden', pb: summary.available ? '22px' : undefined,
              bgcolor: selected ? 'action.selected' : 'background.paper',
              outline: selected ? (theme) => `2px solid ${theme.palette.dashboard.selectionAccent}` : 'none', outlineOffset: -1,
              '&:hover': { borderColor: 'dashboard.controlBorder', filter: 'brightness(0.98)' },
            }}>
            <Typography component="span" sx={{ fontFamily: sansFont, fontSize: summary.available ? fontSizes.section : fontSizes.body, fontWeight: fontWeights.emphasized,
              color: !summary.available ? 'text.muted' : tone === 'neutral' ? 'text.secondary' : `${tone}.main` }}>
              {summary.available && tone !== 'neutral' && <Box component="span" aria-hidden="true" sx={{ mr: '6px', fontSize: '0.7em', verticalAlign: '0.1em' }}>{summary.deltaSeconds > 0 ? '▲' : '▼'}</Box>}{value}
            </Typography>
            <Typography component="span" variant="caption" color="text.secondary">{summary.available ? `${direction} · ${summary.matched} matched` : summary.reason}</Typography>
            {summary.available && summary.deltaPercent === null && <Typography component="span" variant="caption" color="text.secondary">{comparison.baselineDuration === 0 ? 'Percentage unavailable: zero reference runtime' : 'Percentage unavailable'}</Typography>}
            {summary.available && Number.isFinite(summary.deltaPercent) && summary.deltaPercent !== 0 && <Box aria-hidden="true" data-testid="change-magnitude" sx={{ position: 'absolute', left: 16, bottom: 10, height: 3, borderRadius: '2px',
              width: `calc((100% - 32px) * ${Math.min(1, Math.max(0.06, Math.abs(summary.deltaPercent) / MAGNITUDE_FULL_SCALE_PERCENT))})`,
              bgcolor: tone === 'neutral' ? 'dashboard.controlBorder' : `${tone}.main` }} />}
          </ButtonBase>;
        })}
      </Fragment>)}
    </Box>

  </Box>;
}
