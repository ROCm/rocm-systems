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
  Typography,
} from '@mui/material';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import DetailItem from '../shared/DetailItem';
import { DetailGrid, DetailSectionHeading } from '../shared/DetailLayout';
import StatusChip from '../shared/StatusChip';
import { commitTimestampFor } from '../../data/runOrdering';
import { problemDetails } from '../../data/problemDetails';
import { provenanceDetails } from '../../data/provenance';
import { formatDuration, formatFullDate, shortSha } from '../../utils/formatters';
import { hasDisplayValue } from '../../utils/values';

export default function BenchmarkResultDialog({ record, repository, onClose }) {
  const run = record?.run;
  const test = record?.test;
  const provenance = run?.provenance ?? {};
  const commitSha = provenance.rocjitsuCommitSha;
  const testProblemDetails = problemDetails(test?.problem);
  const environmentDetails = provenanceDetails(provenance);

  return (
    <Dialog open={Boolean(record)} onClose={onClose} fullWidth maxWidth="md">
      {record && (
        <>
          <DialogTitle component="div" sx={{ pr: 7 }}>
            <Typography variant="h2" sx={{ lineHeight: '24px' }}>{test.name}</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.4, lineHeight: '21px' }}>{test.target} · {test.suite}</Typography>
            <IconButton onClick={onClose} aria-label="Close details" sx={{ position: 'absolute', top: 11, right: 11 }}><CloseRoundedIcon /></IconButton>
          </DialogTitle>
          <DialogContent dividers>
            <DetailSectionHeading>Result</DetailSectionHeading>
            <DetailGrid>
              <DetailItem label="Status"><StatusChip status={test.status} /></DetailItem>
              <DetailItem label="Duration">{formatDuration(test.durationSeconds)}</DetailItem>
              {test.error && <Box sx={{ gridColumn: '1 / -1' }}><DetailItem label="Error">{test.error}</DetailItem></Box>}
            </DetailGrid>

            <Divider sx={{ my: 2.5 }} />
            <DetailSectionHeading>Problem Details</DetailSectionHeading>
            {testProblemDetails.length > 0 ? (
              <DetailGrid>
                {testProblemDetails.map((detail) => (
                  <DetailItem key={detail.key} label={detail.label}>{detail.value}</DetailItem>
                ))}
              </DetailGrid>
            ) : (
              <Typography variant="body2" color="text.secondary">No problem details were provided for this result.</Typography>
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
              <DetailItem label="Run time">{formatFullDate(run.timestamp)}</DetailItem>
              <DetailItem label="Run type">{run.trigger === 'manual' ? 'Manual' : 'Auto'}</DetailItem>
              {hasDisplayValue(run.machineId) && <DetailItem label="Machine">{run.machineId}</DetailItem>}
              <DetailItem label="RocJitsu commit">{commitSha}</DetailItem>
              <DetailItem label="Commit time">{formatFullDate(commitTimestampFor(run))}</DetailItem>
              {hasDisplayValue(provenance.commitMessage) && <DetailItem label="Commit message">{provenance.commitMessage}</DetailItem>}
              {hasDisplayValue(run.plugin?.name) && <DetailItem label="Plugin">{run.plugin.name}</DetailItem>}
              {hasDisplayValue(run.plugin?.version) && <DetailItem label="Plugin version">{run.plugin.version}</DetailItem>}
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
