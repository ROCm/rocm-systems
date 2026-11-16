import { Box } from '@mui/material';
import { fontWeights } from '../../theme/typography';

// The one expandable-section style: top rule, list-item marker, emphasized title.
export default function Disclosure({ title, children, sx, ...props }) {
  return <Box component="details" {...props} sx={[{ mt: '10px', padding: '10px 2px 4px', borderTop: 1, borderColor: 'divider' }, ...(Array.isArray(sx) ? sx : [sx])]}>
    <Box component="summary" sx={{ cursor: 'pointer', minHeight: { xs: 44, sm: 'auto' }, display: 'list-item', fontWeight: fontWeights.emphasized, ml: 2, mb: '6px' }}>{title}</Box>
    {children}
  </Box>;
}
