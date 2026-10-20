import { Chip } from '@mui/material';
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded';
import ErrorRoundedIcon from '@mui/icons-material/ErrorRounded';
import TimerOffRoundedIcon from '@mui/icons-material/TimerOffRounded';

const statusConfig = {
  completed: { label: 'Completed', color: 'success', icon: <CheckCircleRoundedIcon /> },
  failed: { label: 'Failed', color: 'error', icon: <ErrorRoundedIcon /> },
  timeout: { label: 'Timeout', color: 'warning', icon: <TimerOffRoundedIcon /> },
};

export default function StatusChip({ status, size = 'small' }) {
  const config = statusConfig[status] ?? { label: status || 'Unknown', color: 'default' };
  return (
    <Chip
      size={size}
      variant="outlined"
      label={config.label}
      color={config.color}
      icon={config.icon}
      sx={{ fontWeight: 650, '& .MuiChip-icon': { fontSize: 15 } }}
    />
  );
}
