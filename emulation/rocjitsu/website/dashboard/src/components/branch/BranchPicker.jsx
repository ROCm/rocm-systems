import { Box, IconButton, Typography } from '@mui/material';
import ChevronLeftRounded from '@mui/icons-material/ChevronLeftRounded';
import ChevronRightRounded from '@mui/icons-material/ChevronRightRounded';

// Keep the picker mounted: collapsing is local presentation, not pair selection.
export default function BranchPicker({ collapsed, onToggle, children }) {
  return <Box className="branch-picker" data-picker-collapsed={collapsed} sx={{ minWidth: 0, position: 'relative', display: 'flex',
    '@container branch-runs (min-width:1000px)': {
      '&[data-picker-collapsed="true"] .branch-picker-content': { display: 'none' },
      '&[data-picker-collapsed="true"]': { border: 1, borderColor: 'divider', borderRadius: 1, bgcolor: 'background.paper' },
    },
  }}>
    <IconButton aria-label={collapsed ? 'Expand branches' : 'Collapse branches'} aria-expanded={!collapsed} aria-controls="branch-picker"
      onClick={onToggle} sx={{ display: 'none', '@container branch-runs (min-width:1000px)': { display: 'inline-flex' }, position: 'absolute', top: 4, right: 0, width: 44, height: 44, zIndex: 1,
        '&.Mui-focusVisible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: -2 },
      }}>
      {collapsed ? <ChevronRightRounded /> : <ChevronLeftRounded />}
    </IconButton>
    <Typography aria-hidden="true" variant="caption" sx={{ display: 'none', '@container branch-runs (min-width:1000px)': { display: collapsed ? 'block' : 'none' }, writingMode: 'vertical-rl', mt: 7, mx: 'auto', color: 'text.secondary' }}>Branches</Typography>
    <Box id="branch-picker" className="branch-picker-content" sx={{ display: 'flex', minWidth: 0, width: '100%', '& .branch-list': { width: '100%' }, '@container branch-runs (min-width:1000px)': { '& #branches-heading': { pr: 4 } } }}>{children}</Box>
  </Box>;
}
