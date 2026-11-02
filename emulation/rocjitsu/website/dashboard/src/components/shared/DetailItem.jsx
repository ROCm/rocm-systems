import { Box, Typography } from '@mui/material';

export default function DetailItem({ label, children }) {
  return (
    <Box>
      <Typography
        component="div"
        variant="overline"
        sx={{ color: 'text.secondary', fontSize: '11px', lineHeight: '20px' }}
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
