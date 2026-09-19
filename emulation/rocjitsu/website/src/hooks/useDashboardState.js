import { useMemo, useState } from 'react';

export function useDashboardState(data) {
  const defaultTarget = data.targets.includes('gfx1250') ? 'gfx1250' : data.targets[0];
  const [targets, setTargets] = useState(defaultTarget ? [defaultTarget] : []);
  const [suites, setSuites] = useState(data.suites);
  const [historyRange, setHistoryRange] = useState('ALL');
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [comparisonBaselineId, setComparisonBaselineId] = useState(null);
  const [comparisonCandidateId, setComparisonCandidateId] = useState(null);
  const [explorerRunIds, setExplorerRunIds] = useState([]);

  const filters = useMemo(() => ({ targets, suites }), [targets, suites]);

  return {
    filters,
    targets,
    suites,
    historyRange,
    tab,
    search,
    comparisonBaselineId,
    comparisonCandidateId,
    explorerRunIds,
    setTargets,
    setSuites,
    setHistoryRange,
    setTab,
    setSearch,
    setComparisonBaselineId,
    setComparisonCandidateId,
    setExplorerRunIds,
  };
}
