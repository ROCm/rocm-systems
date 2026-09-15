import {
  AppBar,
  Box,
  Button,
  Chip,
  IconButton,
  Stack,
  Toolbar,
  Tooltip,
  Typography,
} from '@mui/material';
import DarkModeRoundedIcon from '@mui/icons-material/DarkModeRounded';
import LightModeRoundedIcon from '@mui/icons-material/LightModeRounded';
import DownloadRoundedIcon from '@mui/icons-material/DownloadRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import FiberManualRecordRoundedIcon from '@mui/icons-material/FiberManualRecordRounded';
import { formatFullDate, formatRelativeTime } from '../../utils/formatters';

export default function DashboardHeader({ data, downloadData, loading = false, mode, onToggleMode }) {
  const downloadJson = () => {
    if (!downloadData) return;
    const blob = new Blob([JSON.stringify(downloadData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'rocjitsu-benchmark-data.json';
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <AppBar
      position="sticky"
      color="inherit"
      elevation={0}
      sx={{
        borderBottom: 1,
        borderColor: 'divider',
        backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(21,28,43,.92)' : 'rgba(255,255,255,.9)',
        backdropFilter: 'blur(16px)',
        '&::after': {
          content: '""',
          position: 'absolute',
          inset: 'auto 0 -1px 0',
          height: 2,
          background: 'linear-gradient(90deg, #5566E9, #8A5DE8 42%, #12A594 78%, transparent)',
        },
      }}
    >
      <Toolbar sx={{ width: '100%', maxWidth: 1600, mx: 'auto', minHeight: { xs: 68, md: 76 }, px: { xs: 2, sm: 3, xl: 4 } }}>
        <Stack direction="row" sx={{ alignItems: 'center', gap: 1.4, minWidth: 0, flex: 1 }}>
          <Box sx={{ minWidth: 0 }}>
            <Stack direction="row" sx={{ alignItems: 'center', gap: 0.8, flexWrap: 'wrap' }}>
              <Typography sx={{ fontSize: { xs: 16, sm: 19 }, fontWeight: 780, letterSpacing: '-.025em', lineHeight: 1.15 }}>
                RocJitsu <Box component="span" sx={{ color: 'text.disabled', fontWeight: 400, mx: 0.45 }}>/</Box>{' '}
                <Box component="span" sx={{ color: 'text.secondary', fontWeight: 580 }}>Performance Dashboard</Box>
              </Typography>
              {data?.isBeta && (
                <Chip size="small" label="Beta" color="warning" sx={{ height: 19, fontSize: 9, fontWeight: 800, textTransform: 'uppercase' }} />
              )}
            </Stack>
            <Stack direction="row" sx={{ alignItems: 'center', gap: 0.6, mt: 0.45, color: 'text.secondary' }}>
              <FiberManualRecordRoundedIcon
                color={loading ? 'primary' : 'success'}
                sx={loading ? {
                  fontSize: 9,
                  animation: 'dataLoadingPulse 1.25s ease-in-out infinite',
                  '@keyframes dataLoadingPulse': {
                    '0%, 100%': { opacity: 0.35, transform: 'scale(.8)' },
                    '50%': { opacity: 1, transform: 'scale(1.25)' },
                  },
                } : { fontSize: 9 }}
              />
              {loading ? (
                <Typography variant="caption">Loading run data…</Typography>
              ) : data?.generatedAt ? (
                <Tooltip title={formatFullDate(data.generatedAt)}>
                  <Typography variant="caption">Updated {formatRelativeTime(data.generatedAt)}</Typography>
                </Tooltip>
              ) : (
                <Typography variant="caption">Data unavailable</Typography>
              )}
              {data?.repository && (
                <>
                  <Typography variant="caption" sx={{ color: 'text.disabled' }}>·</Typography>
                  <Typography
                    component="a"
                    href={`${data.repository}/tree/develop/emulation/rocjitsu`}
                    target="_blank"
                    rel="noreferrer"
                    variant="caption"
                    color="primary.main"
                    sx={{ textDecoration: 'none', display: { xs: 'none', sm: 'inline-flex' }, alignItems: 'center', gap: 0.4 }}
                  >
                    ROCm/rocm-systems <OpenInNewRoundedIcon sx={{ fontSize: 11 }} />
                  </Typography>
                </>
              )}
            </Stack>
          </Box>
        </Stack>
        <Stack direction="row" sx={{ gap: 0.75, ml: 1 }}>
          <Button
            variant="outlined"
            color="inherit"
            startIcon={<DownloadRoundedIcon />}
            disabled={!downloadData}
            onClick={downloadJson}
            sx={{ display: { xs: 'none', sm: 'inline-flex' }, borderColor: 'divider' }}
          >
            Download JSON
          </Button>
          <Tooltip title={`Use ${mode === 'dark' ? 'light' : 'dark'} theme`}>
            <IconButton aria-label={`Use ${mode === 'dark' ? 'light' : 'dark'} theme`} onClick={onToggleMode} sx={{ border: 1, borderColor: 'divider', borderRadius: 2.2 }}>
              {mode === 'dark' ? <LightModeRoundedIcon /> : <DarkModeRoundedIcon />}
            </IconButton>
          </Tooltip>
        </Stack>
      </Toolbar>
    </AppBar>
  );
}
