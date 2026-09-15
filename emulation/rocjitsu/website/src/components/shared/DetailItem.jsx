import { Box, Typography } from '@mui/material';

export default function DetailItem({ label, children, wide = false }) {
  return (
    <Box sx={wide ? { gridColumn: '1 / -1' } : undefined}>
      <Typography
        component="div"
        variant="overline"
        color="text.secondary"
        sx={{ fontSize: '11px', lineHeight: '20px' }}
      >
        {label}
      </Typography>
      <Typography
        variant="body2"
        component="div"
        sx={{ mt: 0.25, lineHeight: '21px', overflowWrap: 'anywhere' }}
      >
        {children ?? '—'}
      </Typography>
    </Box>
  );
}
