import { fontSizes, fontWeights, textStyles } from '../../theme/typography';
import { Box, IconButton, Stack, Typography } from '@mui/material';
import SurfaceButton from '../shared/SurfaceButton';
import HoverTooltip from '../shared/HoverTooltip';
import { shapeTokens } from '../../theme/tokens';
import DarkModeRoundedIcon from '@mui/icons-material/DarkModeRounded';
import LightModeRoundedIcon from '@mui/icons-material/LightModeRounded';
import DownloadRoundedIcon from '@mui/icons-material/DownloadRounded';
import CloudSyncRoundedIcon from '@mui/icons-material/CloudSyncRounded';
import GitHubIcon from '@mui/icons-material/GitHub';
import { formatPublicationTime } from '../../utils/formatters';
import { repositoryLink as buildRepositoryLink } from '../../utils/repositoryLink';
import { dashboardPageLabels } from '../../config/dashboardPages';

// Top-right actions: circular icon buttons and a pill link, one 40px height.
const actionStyles = { width: { xs: 44, sm: 40 }, height: { xs: 44, sm: 40 }, border: 1, borderColor: 'dashboard.controlBorder', borderRadius: '50%', color: 'text.secondary', '&:hover': { bgcolor: 'action.hover' }, '& svg': { fontSize: 22 } };

export default function DashboardHeader({ data, dataError = null, downloadData, loading = false, mode, onReloadData, onToggleMode, tab = 'overview' }) {
  const publicationLabel = formatPublicationTime(data?.generatedAt);
  const repositoryLink = buildRepositoryLink(data?.repository, data?.canonicalBranch);
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
    <Box component="header" sx={{ borderBottom: 1, borderColor: 'dashboard.border', bgcolor: 'background.paper', px: { xs: 2, md: 3 }, py: shapeTokens.headerPadding }}>
      <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
        <Box sx={{ minWidth: 0 }}>
          <Typography sx={{ ...textStyles.captionCompact, color: 'text.secondary', textTransform: 'uppercase', letterSpacing: '0.06em' }}>rocjitsu / {dashboardPageLabels[tab]}</Typography>
          <Typography component="h1" variant="h1" sx={{ mt: '2px' }}>{dashboardPageLabels[tab]}</Typography>
            <Typography variant="caption" color="text.muted" sx={{ display: 'block', mt: '4px' }}>
              {loading ? 'Loading run data…' : dataError ? 'Data unavailable' : publicationLabel ? <>Data as of <Box component="strong" sx={{ fontWeight: fontWeights.emphasized, color: 'text.primary' }}><time dateTime={data.generatedAt}>{publicationLabel}</time></Box></> : 'Data unavailable'}
            </Typography>
        </Box>
        <Stack direction="row" sx={{ gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          {repositoryLink && <SurfaceButton component="a" href={repositoryLink.href} target="_blank" rel="noreferrer"
            startIcon={<GitHubIcon />} sx={{ height: { xs: 44, sm: 40 }, px: '16px', borderRadius: '999px', borderColor: 'dashboard.controlBorder', fontSize: fontSizes.body }}>{repositoryLink.label}</SurfaceButton>}
          <HoverTooltip title="Reload all data (bypass cached files)"><IconButton aria-label="Reload all data" color={dataError ? 'primary' : 'default'} disabled={loading} onClick={onReloadData} sx={actionStyles}><CloudSyncRoundedIcon /></IconButton></HoverTooltip>
          <HoverTooltip title="Download JSON"><IconButton aria-label="Download JSON" disabled={loading || !downloadData} onClick={downloadJson} sx={actionStyles}><DownloadRoundedIcon /></IconButton></HoverTooltip>
          <HoverTooltip title={`Use ${mode === 'dark' ? 'light' : 'dark'} theme`}><IconButton aria-label={`Use ${mode === 'dark' ? 'light' : 'dark'} theme`} onClick={onToggleMode} sx={actionStyles}>{mode === 'dark' ? <LightModeRoundedIcon /> : <DarkModeRoundedIcon />}</IconButton></HoverTooltip>
        </Stack>
      </Stack>
    </Box>
  );
}
