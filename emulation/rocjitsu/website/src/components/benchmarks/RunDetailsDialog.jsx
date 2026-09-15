import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  IconButton,
  Link,
  Stack,
  Typography,
} from '@mui/material';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import DetailItem from '../shared/DetailItem';
import { DetailGrid, DetailSectionHeading } from '../shared/DetailLayout';
import StatusChip from '../shared/StatusChip';
import { commitTimestampFor } from '../../data/runOrdering';
import { provenanceDetails } from '../../data/provenance';
import { formatDuration, formatFullDate, shortSha } from '../../utils/formatters';
import { hasDisplayValue } from '../../utils/values';

export default function RunDetailsDialog({ run, filters, repository, onClose }) {
  const selectedTests = (run?.tests ?? []).filter((test) => (
    filters.targets.includes(test.target) && filters.suites.includes(test.suite)
  ));
  const completedTests = selectedTests.filter((test) => (
    test.status === 'completed' && Number.isFinite(test.durationSeconds)
  ));
  const incompleteTests = selectedTests.filter((test) => (
    test.status !== 'completed' || !Number.isFinite(test.durationSeconds)
  ));
  const totalTestCount = selectedTests.length;
  const complete = totalTestCount > 0 && completedTests.length === totalTestCount;
  const duration = complete
    ? completedTests.reduce((total, test) => total + test.durationSeconds, 0)
    : null;
  const provenance = run?.provenance ?? {};
  const commitSha = provenance.rocjitsuCommitSha;
  const environmentDetails = provenanceDetails(provenance);

  return (
    <Dialog open={Boolean(run)} onClose={onClose} fullWidth maxWidth="sm">
      {run && (
        <>
          <DialogTitle component="div" sx={{ pr: 7 }}>
            <Typography variant="h2" sx={{ lineHeight: '24px' }}>Run Details</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.4, lineHeight: '21px' }}>
              Commit {shortSha(run)} · {run.trigger === 'manual' ? 'Manual' : 'Auto'}
            </Typography>
            <IconButton onClick={onClose} aria-label="Close Run Details" sx={{ position: 'absolute', top: 11, right: 11 }}>
              <CloseRoundedIcon />
            </IconButton>
          </DialogTitle>
          <DialogContent dividers>
            <DetailSectionHeading>Selected Scope</DetailSectionHeading>
            <DetailGrid>
              <DetailItem label="Coverage">{completedTests.length}/{totalTestCount} completed</DetailItem>
              <DetailItem label="Total duration">{formatDuration(duration)}</DetailItem>
              <DetailItem label="Run time">{formatFullDate(run.timestamp)}</DetailItem>
              <DetailItem label="Commit time">{formatFullDate(commitTimestampFor(run))}</DetailItem>
            </DetailGrid>

            {incompleteTests.length > 0 && (
              <>
                <Divider sx={{ my: 2.5 }} />
                <DetailSectionHeading>Incomplete Tests</DetailSectionHeading>
                <Stack spacing={1.25}>
                  {incompleteTests.map((test) => (
                    <Box
                      key={test.testId}
                      sx={{
                        border: 1,
                        borderColor: 'divider',
                        borderRadius: 1.5,
                        px: 1.5,
                        py: 1.25,
                      }}
                    >
                      <Stack direction="row" spacing={1.5} sx={{ alignItems: 'flex-start', justifyContent: 'space-between' }}>
                        <Box sx={{ minWidth: 0 }}>
                          <Typography variant="body2" sx={{ fontWeight: 700 }}>{test.name}</Typography>
                          <Typography variant="caption" color="text.secondary">{test.target} · {test.suite}</Typography>
                        </Box>
                        <StatusChip status={test.status} />
                      </Stack>
                      {hasDisplayValue(test.error) && (
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>{test.error}</Typography>
                      )}
                    </Box>
                  ))}
                </Stack>
              </>
            )}

            <Divider sx={{ my: 2.5 }} />
            <DetailSectionHeading>Environment</DetailSectionHeading>
            {environmentDetails.length > 0 ? (
              <DetailGrid>
                {environmentDetails.map((detail) => (
                  <DetailItem key={detail.key} label={detail.label}>{detail.value}</DetailItem>
                ))}
              </DetailGrid>
            ) : (
              <Typography variant="body2" color="text.secondary">No environment details were provided for this run.</Typography>
            )}

            <Divider sx={{ my: 2.5 }} />
            <DetailSectionHeading>Run Provenance</DetailSectionHeading>
            <DetailGrid>
              <DetailItem label="RocJitsu commit">{commitSha}</DetailItem>
              {hasDisplayValue(provenance.commitMessage) && <DetailItem label="Commit message">{provenance.commitMessage}</DetailItem>}
              {hasDisplayValue(run.plugin?.name) && <DetailItem label="Plugin">{run.plugin.name}</DetailItem>}
              {hasDisplayValue(run.plugin?.version) && <DetailItem label="Plugin version">{run.plugin.version}</DetailItem>}
              {hasDisplayValue(run.machineId) && <DetailItem label="Machine">{run.machineId}</DetailItem>}
              {hasDisplayValue(run.branch) && <DetailItem label="Branch">{run.branch}</DetailItem>}
            </DetailGrid>
          </DialogContent>
          <DialogActions sx={{ px: 3, py: 1.5 }}>
            {repository && commitSha && (
              <Button
                component={Link}
                href={`${repository}/commit/${commitSha}`}
                target="_blank"
                rel="noreferrer"
                endIcon={<OpenInNewRoundedIcon />}
                sx={{ mr: 'auto' }}
              >
                Commit {shortSha(run)}
              </Button>
            )}
            <Button onClick={onClose} color="inherit">Close</Button>
          </DialogActions>
        </>
      )}
    </Dialog>
  );
}
