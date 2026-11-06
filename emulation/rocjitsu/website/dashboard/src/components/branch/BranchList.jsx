import { fontWeights, sansFont } from '../../theme/typography';
import ActionButton from '../shared/ActionButton';
import { Box, ButtonBase, Paper, Typography } from '@mui/material';
import { formatFullDate, shortSha } from '../../utils/formatters';

export default function BranchList({ branches, selectedBranch, onSelect, onClear, scrollRef, onScroll }) {
  return <Paper variant="outlined" className="branch-list" data-testid="branch-list-panel" component="section" aria-labelledby="branches-heading" sx={{
    display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0,
    '@container branch-runs (min-width:1000px)': { flex: 1, minHeight: 0 },
    '@media (min-width:1280px)': { alignSelf: 'stretch', minHeight: 0 },
  }}>
    <Box sx={{ padding: '20px 20px 0', mb: '10px', flexShrink: 0 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1 }}>
        <Typography id="branches-heading" component="h2" variant="subtitle1">Branches</Typography>
      </Box>
      <Typography variant="caption" color="text.secondary">Newest published executions first</Typography>
    </Box>
    <Box ref={scrollRef} onScroll={onScroll} tabIndex={0} role="region" aria-label="Published branches, scrollable" data-testid="branch-list-scroll" sx={{
      overflowY: 'auto', overscrollBehavior: 'contain', scrollbarGutter: 'stable',
      maxHeight: 'min(60vh, 560px)', minHeight: 180,
      '@media (min-width:1280px)': { maxHeight: 560 },
      '@container branch-runs (min-width:1000px)': { height: 0, minHeight: 180, maxHeight: 'none', flex: '1 1 0' },
      '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: -2 },
    }}>
      {branches.length ? branches.map((entry) => <ButtonBase key={entry.branch} onClick={() => onSelect(entry)}
        aria-label={`View branch ${entry.branch}${entry.pullRequest ? `, PR #${entry.pullRequest.number}` : ', no PR'}`}
        aria-pressed={selectedBranch === entry.branch} aria-controls="branch-detail" data-testid={`branch-row-${entry.branch}`} sx={{
          width: '100%', display: 'block', textAlign: 'left', padding: '8px 20px', lineHeight: '20px',
          borderTop: 1, borderColor: 'divider', color: 'text.primary',
          bgcolor: selectedBranch === entry.branch ? 'action.selected' : 'transparent',
          boxShadow: selectedBranch === entry.branch ? (theme) => `inset 3px 0 0 ${theme.palette.dashboard.selectionAccent}` : 'none',
          '&:hover': { bgcolor: selectedBranch === entry.branch ? 'action.selected' : 'action.hover' },
          '&.Mui-focusVisible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: -2 },
        }}>
        <Typography variant="body2" color="text.primary" sx={{ fontWeight: fontWeights.emphasized, overflowWrap: 'anywhere' }}>{entry.branch}</Typography>
        <Typography component="div" variant="caption" color="text.primary" sx={{ mt: '4px' }}>
          <Box component="span" sx={{ fontFamily: sansFont, fontWeight: fontWeights.emphasized }}>{shortSha(entry.latestRun)}</Box>
          {entry.pullRequest ? ` · PR #${entry.pullRequest.number}` : ' · No PR'}
        </Typography>
        <Typography component="div" variant="caption" color="text.muted" sx={{ mt: '2px' }}>{formatFullDate(entry.latestRun.timestamp)}</Typography>
      </ButtonBase>) : <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '4px' }}>
        <Typography variant="body2">No matching branches</Typography>
        <Typography variant="caption" color="text.secondary">Try another branch, PR or SHA.</Typography>
        <ActionButton onClick={onClear} sx={{ mt: 1, minHeight: 44, px: 0 }}>Clear filters</ActionButton>
      </Box>}
    </Box>
    <Typography variant="caption" color="text.muted" sx={{ padding: '10px 20px', borderTop: 1, borderColor: 'divider', flexShrink: 0 }}>
      Only branches active within 30 days of the latest published data are shown.
    </Typography>
  </Paper>;
}
