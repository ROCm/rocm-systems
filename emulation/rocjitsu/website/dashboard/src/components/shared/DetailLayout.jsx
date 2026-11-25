import { Box, Typography } from '@mui/material';

export function DetailGrid({ children }) {
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))' }, gap: '18px' }}>
      {children}
    </Box>
  );
}

export function DetailSectionHeading({ children }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: '12px' }}>
      <Box aria-hidden sx={{ width: 3, height: 20, flexShrink: 0, borderRadius: 2, bgcolor: 'primary.main' }} />
      <Typography component="h3" variant="h3">
        {children}
      </Typography>
    </Box>
  );
}
