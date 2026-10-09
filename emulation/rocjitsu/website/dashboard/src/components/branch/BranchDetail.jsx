import { useRef, useState } from 'react';
import { Box, Button, Paper, Typography } from '@mui/material';
import { selectConfigurationComparison } from '../../data/branchSelectors';
import CategoryTag from '../shared/CategoryTag';
import BranchRunSelector from './BranchRunSelector';
import ConfigurationMatrix from './ConfigurationMatrix';
import BenchmarkDifferences from './BenchmarkDifferences';
import BranchEnvironment from './BranchEnvironment';
import { configurationState, safeExternalUrl } from './branchPresentation';

export default function BranchDetail({ selection, candidate, reference, automatic, candidateOptions, runs, onCandidate, onReference, onAutomatic, onBack, onConfiguration, onOpenComparison, repository, headingRef }) {
  const differencesRef = useRef(null);
  const [selectedSuites, setSelectedSuites] = useState(null);
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState({ key: 'seconds', direction: 'desc' });
  const suiteOptions = [...new Set([...(candidate?.tests ?? []), ...(reference?.tests ?? [])].map((test) => test.suite).concat(selectedSuites ?? []))].sort();
  const suites = selectedSuites ?? suiteOptions;
  const comparison = selectConfigurationComparison(candidate, reference, { target: selection.target, mode: selection.mode, suites, query });
  const summary = configurationState(candidate, reference, comparison, selection.target, selection.mode);
  return <Paper variant="outlined" component="section" id="branch-detail" className="branch-detail" aria-labelledby="branch-detail-heading" sx={{ minWidth: 0, p: { xs: 2, sm: 2.5 }, overflowWrap: 'anywhere' }}>
    <Button onClick={onBack} className="branch-back" sx={{ display: 'none', '@media (max-width:767px)': { display: 'inline-flex' }, minHeight: 44, mb: 1 }}>← Back to branches</Button>
    <Box sx={{ pb: 2, mb: 2.5, borderBottom: 1, borderColor: 'divider' }}>
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography id="branch-detail-heading" ref={headingRef} component="h2" variant="h6" tabIndex={-1} sx={{ fontWeight: 600, '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: 4 } }}>{selection.branch}</Typography>
        {safeExternalUrl(candidate?.pullRequest?.url) ? <Button component="a" href={candidate.pullRequest.url} target="_blank" rel="noopener noreferrer" sx={{ minHeight: 44 }}>PR #{candidate.pullRequest.number} ↗</Button> : <Typography variant="caption" color="text.secondary">{candidate?.pullRequest ? `PR #${candidate.pullRequest.number} · link not provided` : 'No PR'}</Typography>}
      </Box>
      {candidate?.provenance?.commitMessage && <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75 }}>{candidate.provenance.commitMessage}</Typography>}
    </Box>
    <Box sx={{ display: 'grid', gap: 2 }}>
      <BranchRunSelector label="Candidate" options={candidateOptions} value={candidate} selectedId={selection.candidateId} onChange={onCandidate} />
      <BranchRunSelector label="Reference" options={runs} value={reference} selectedId={selection.referenceId} onChange={onReference} />
    </Box>
    {!candidate && <Typography variant="body2" sx={{ mb: 2 }}>Selected candidate is not in the current published data. Its exact identity is retained; choose another published attempt above.</Typography>}
    <Typography variant="body2" color="text.secondary" data-testid="branch-reference-reason" sx={{ mt: 1.5 }}>
      {selection.referenceId && !reference ? 'Saved reference attempt is unavailable. Its exact identity is retained; choose a published reference above.' : selection.manual ? 'Manual reference · any published attempt; only matching target, mode and completed workloads are compared.' : reference && reference.runId !== automatic.run?.runId ? 'Saved automatic reference attempt retained. Only matching completed workloads are compared.' : automatic.description}
    </Typography>
    {selection.manual && <Button onClick={onAutomatic} sx={{ minHeight: 44 }}>Restore automatic base</Button>}
    <ConfigurationMatrix runs={runs} candidate={candidate} reference={reference} selection={selection} suites={suites} query={query} onSelect={(target, mode) => {
      onConfiguration(target, mode);
      differencesRef.current?.focus({ preventScroll: true });
    }} />
    <Box ref={differencesRef} id="branch-benchmark-differences" role="region" tabIndex={-1} aria-labelledby="branch-differences-heading" sx={{ mt: 2.5, pt: 2, borderTop: 1, borderColor: 'divider', '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: 4 } }}>
      <Box sx={{ display: 'flex', gap: 1, justifyContent: 'space-between', flexWrap: 'wrap', alignItems: 'center' }}>
        <Typography id="branch-differences-heading" component="h3" variant="subtitle1" sx={{ fontWeight: 600 }}>Benchmark differences</Typography>
        <Box data-testid="branch-selected-configuration" sx={{ display: 'flex', gap: 0.5 }}><CategoryTag kind="target" label={selection.target} /><CategoryTag kind="mode" label={selection.mode} /></Box>
      </Box>
      <BenchmarkDifferences comparison={comparison} summary={summary} candidate={candidate} reference={reference} selection={selection} suites={suites} suiteOptions={suiteOptions} query={query} sort={sort}
        onSuites={setSelectedSuites} onQuery={setQuery} onSort={setSort} onClear={() => { setSelectedSuites(null); setQuery(''); }} onOpenComparison={onOpenComparison} />
    </Box>
    <BranchEnvironment candidate={candidate} reference={reference} selection={selection} repository={repository} />
  </Paper>;
}
