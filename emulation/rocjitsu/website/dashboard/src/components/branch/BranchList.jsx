import { Box, Button, ButtonBase, Paper, Typography } from '@mui/material';
import { formatFullDate, shortSha } from '../../utils/formatters';
import { monoFont } from '../../theme/tokens';

export default function BranchList({ branches, selectedBranch, onSelect, onClear, scrollRef, onScroll }) {
  return <Paper variant="outlined" className="branch-list" data-testid="branch-list-panel" component="section" aria-labelledby="branches-heading" sx={{
    display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0,
    '@media (min-width:1280px)': { alignSelf: 'stretch', minHeight: 0 },
  }}>
    <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1 }}>
        <Typography id="branches-heading" component="h2" variant="subtitle1" sx={{ fontWeight: 600 }}>Branches</Typography>
      </Box>
      <Typography variant="caption" color="text.secondary">Newest published executions first</Typography>
    </Box>
    <Box ref={scrollRef} onScroll={onScroll} tabIndex={0} role="region" aria-label="Published branches, scrollable" data-testid="branch-list-scroll" sx={{
      overflowY: 'auto', overscrollBehavior: 'contain', scrollbarGutter: 'stable',
      maxHeight: 'min(60vh, 480px)', minHeight: 180,
      '@media (min-width:768px) and (max-width:1279px)': { height: 280, maxHeight: 360 },
      '@media (min-width:1280px)': { height: 0, minHeight: 280, maxHeight: 'none', flex: '1 1 0' },
      '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: -2 },
    }}>
      {branches.length ? branches.map((entry) => <ButtonBase key={entry.branch} onClick={() => onSelect(entry)}
        aria-label={`View branch ${entry.branch}${entry.pullRequest ? `, PR #${entry.pullRequest.number}` : ', no PR'}`}
        aria-pressed={selectedBranch === entry.branch} aria-controls="branch-detail" data-testid={`branch-row-${entry.branch}`} sx={{
          width: '100%', display: 'block', textAlign: 'left', p: 1.5, minHeight: 100,
          borderBottom: 1, borderColor: 'divider', color: 'text.primary',
          bgcolor: selectedBranch === entry.branch ? 'action.selected' : 'transparent',
          '&:hover': { bgcolor: 'action.hover' },
          '&.Mui-focusVisible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: -2 },
        }}>
        <Typography variant="body2" sx={{ fontWeight: 600, overflowWrap: 'anywhere' }}>{entry.branch}</Typography>
        <Typography component="div" variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
          <Box component="span" sx={{ fontFamily: monoFont }}>{shortSha(entry.latestRun)}</Box>
          {entry.pullRequest ? ` · PR #${entry.pullRequest.number}` : ' · No PR'}
        </Typography>
        <Typography component="div" variant="caption" color="text.secondary">{formatFullDate(entry.latestRun.timestamp)}</Typography>
        <Typography component="div" variant="caption" color="text.secondary">{entry.latestRun.configurations.length} published configurations</Typography>
      </ButtonBase>) : <Box sx={{ p: 2 }}>
        <Typography variant="body2">No matching branches</Typography>
        <Typography variant="caption" color="text.secondary">Try another branch, PR or SHA.</Typography>
        <Button onClick={onClear} sx={{ mt: 1, minHeight: 44 }}>Clear filters</Button>
      </Box>}
    </Box>
    <Typography variant="caption" color="text.secondary" sx={{ p: 1.5, borderTop: 1, borderColor: 'divider', flexShrink: 0 }}>
      Only branches active within 30 days of the latest published data are shown.
    </Typography>
  </Paper>;
}
