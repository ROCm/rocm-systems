import { Box, Typography } from '@mui/material';

export function DetailGrid({ children }) {
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))' }, gap: 2.2 }}>
      {children}
    </Box>
  );
}

export function DetailSectionHeading({ children }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
      <Box aria-hidden sx={{ width: 3, height: 20, flexShrink: 0, borderRadius: 2, bgcolor: 'primary.main' }} />
      <Typography component="h3" sx={{ fontSize: 14, lineHeight: '20px', fontWeight: 800, letterSpacing: '-.01em' }}>
        {children}
      </Typography>
    </Box>
  );
}
