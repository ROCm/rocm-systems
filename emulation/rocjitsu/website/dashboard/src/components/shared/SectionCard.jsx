import { Box, Card, CardContent, Stack, Typography } from '@mui/material';

export default function SectionCard({ title, subtitle, action, children, contentSx, sx, ...cardProps }) {
  return (
    <Card {...cardProps} sx={[{ minWidth: 0 }, ...(Array.isArray(sx) ? sx : [sx])]}>
      <CardContent sx={[{ p: { xs: 2, sm: 2.5 }, '&:last-child': { pb: { xs: 2, sm: 2.5 } } }, ...(Array.isArray(contentSx) ? contentSx : [contentSx])]}>
        {(title || action) && (
          <Stack direction={{ xs: 'column', sm: 'row' }} sx={{ justifyContent: 'space-between', alignItems: { xs: 'stretch', sm: 'flex-start' }, gap: 1.5, mb: 2 }}>
            <Box sx={{ minWidth: 0 }}>
              {title && <Typography component="h2" variant="h2">{title}</Typography>}
              {subtitle && <Typography component="p" variant="caption" sx={{ color: 'text.secondary', mt: 0.5 }}>{subtitle}</Typography>}
            </Box>
            {action}
          </Stack>
        )}
        {children}
      </CardContent>
    </Card>
  );
}
