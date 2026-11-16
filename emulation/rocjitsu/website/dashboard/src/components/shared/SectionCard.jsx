import { Box, Card, CardContent, Stack, Typography } from '@mui/material';
import { shapeTokens } from '../../theme/tokens';

export default function SectionCard({ title, subtitle, action, children, contentSx, sx, ...cardProps }) {
  return (
    <Card {...cardProps} sx={[{ minWidth: 0 }, ...(Array.isArray(sx) ? sx : [sx])]}>
      <CardContent sx={[{ p: shapeTokens.cardPadding, '&:last-child': { pb: shapeTokens.cardPadding } }, ...(Array.isArray(contentSx) ? contentSx : [contentSx])]}>
        {(title || action) && (
          <Stack direction={{ xs: 'column', sm: 'row' }} sx={{ justifyContent: 'space-between', alignItems: { xs: 'stretch', sm: 'flex-start' }, gap: '12px', mb: '14px', pb: shapeTokens.cardHeaderPadding, borderBottom: 1, borderColor: 'divider' }}>
            <Box sx={{ minWidth: 0 }}>
              {title && <Typography component="h2" variant="h2">{title}</Typography>}
              {subtitle && <Typography component="p" variant="subtitle2" sx={{ color: 'text.secondary', mt: '2px' }}>{subtitle}</Typography>}
            </Box>
            {action}
          </Stack>
        )}
        {children}
      </CardContent>
    </Card>
  );
}
