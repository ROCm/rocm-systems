import { Button } from '@mui/material';

// Neutral secondary actions share paint; callers own placement and sizing.
export default function SurfaceButton({ variant = 'outlined', sx, ...props }) {
  return <Button {...props} variant={variant} color="inherit" sx={[
    { bgcolor: 'transparent', color: 'text.secondary', borderColor: 'dashboard.border', '&:hover': { bgcolor: 'action.hover' } },
    ...(Array.isArray(sx) ? sx : [sx]),
  ]} />;
}
