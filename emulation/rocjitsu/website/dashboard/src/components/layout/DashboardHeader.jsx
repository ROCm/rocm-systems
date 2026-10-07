import { Box, Button, IconButton, Stack, Tooltip, Typography } from '@mui/material';
import DarkModeRoundedIcon from '@mui/icons-material/DarkModeRounded';
import LightModeRoundedIcon from '@mui/icons-material/LightModeRounded';
import DownloadRoundedIcon from '@mui/icons-material/DownloadRounded';
import CloudSyncRoundedIcon from '@mui/icons-material/CloudSyncRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import { formatFullDate } from '../../utils/formatters';

const titles = { overview: 'Overview', branch: 'Branch Runs', benchmarks: 'Benchmarks', compare: 'Run Comparison' };
const dateFormatter = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
const timeFormatter = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZone: 'UTC' });
const actionStyles = { width: { xs: 44, sm: 32 }, height: { xs: 44, sm: 32 }, border: 1, borderColor: 'divider', borderRadius: '6px', '& svg': { fontSize: 17 } };

export default function DashboardHeader({ data, dataError = null, downloadData, loading = false, mode, onReloadData, onToggleMode, tab = 'overview' }) {
  const publicationDate = data?.generatedAt ? new Date(data.generatedAt) : null;
  const publicationAvailable = publicationDate && Number.isFinite(publicationDate.getTime());
  const publicationLabel = publicationAvailable ? `${dateFormatter.format(publicationDate)} · ${timeFormatter.format(publicationDate)} UTC` : null;
  const downloadJson = () => {
    if (loading || !downloadData) return;
    const blob = new Blob([JSON.stringify(downloadData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'rocjitsu-simulation-benchmark-data.json';
    anchor.click();
    URL.revokeObjectURL(url);
  };
  return (
    <Box component="header" sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper', px: { xs: 2, md: 3 }, py: { xs: 2, md: 2.5 } }}>
      <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1.5 }}>
        <Box sx={{ minWidth: 0 }}>
          <Typography sx={{ fontSize: 11, lineHeight: '16px', color: 'text.secondary', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Rocjitsu / {titles[tab]}</Typography>
          <Typography component="h1" variant="h1" sx={{ mt: 0.25 }}>{titles[tab]}</Typography>
          <Tooltip title={publicationAvailable ? formatFullDate(data.generatedAt) : ''}>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
              {loading ? 'Loading run data…' : dataError ? 'Data unavailable' : publicationLabel ? <>Data as of <Box component="strong" sx={{ fontWeight: 600, color: 'text.primary' }}><time dateTime={data.generatedAt}>{publicationLabel}</time></Box></> : 'Data unavailable'}
            </Typography>
          </Tooltip>
        </Box>
        <Stack direction="row" sx={{ gap: 0.75, alignItems: 'center', flexWrap: 'wrap' }}>
          {data?.repository && <Button component="a" href={data.repository} target="_blank" rel="noreferrer"
            variant="outlined" color="inherit" startIcon={<OpenInNewRoundedIcon />} sx={{ borderColor: 'divider', height: { xs: 44, sm: 32 }, fontSize: 12, color: 'text.secondary', bgcolor: 'background.default' }}>Repository</Button>}
          <Tooltip title="Reload all data (bypass cached files)"><span><IconButton aria-label="Reload all data" color={dataError ? 'primary' : 'default'} disabled={loading} onClick={onReloadData} sx={actionStyles}><CloudSyncRoundedIcon /></IconButton></span></Tooltip>
          <Tooltip title="Download JSON"><span><IconButton aria-label="Download JSON" disabled={loading || !downloadData} onClick={downloadJson} sx={actionStyles}><DownloadRoundedIcon /></IconButton></span></Tooltip>
          <Tooltip title={`Use ${mode === 'dark' ? 'light' : 'dark'} theme`}><IconButton aria-label={`Use ${mode === 'dark' ? 'light' : 'dark'} theme`} onClick={onToggleMode} sx={actionStyles}>{mode === 'dark' ? <LightModeRoundedIcon /> : <DarkModeRoundedIcon />}</IconButton></Tooltip>
        </Stack>
      </Stack>
    </Box>
  );
}
