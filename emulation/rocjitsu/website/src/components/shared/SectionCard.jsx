import { Box, Card, CardContent, Stack, Typography } from '@mui/material';

export default function SectionCard({ title, subtitle, action, children, contentSx, ...cardProps }) {
  return (
    <Card {...cardProps}>
      <CardContent sx={{ p: { xs: 2, sm: 2.5 }, '&:last-child': { pb: { xs: 2, sm: 2.5 } }, ...contentSx }}>
        {(title || action) && (
          <Stack direction={{ xs: 'column', sm: 'row' }} sx={{ justifyContent: 'space-between', alignItems: { xs: 'stretch', sm: 'flex-start' }, gap: 1.5, mb: 2.25 }}>
            <Box>
              <Typography variant="h2">{title}</Typography>
              {subtitle && <Typography variant="body2" color="text.secondary" sx={{ mt: 0.4 }}>{subtitle}</Typography>}
            </Box>
            {action}
          </Stack>
        )}
        {children}
      </CardContent>
    </Card>
  );
}
