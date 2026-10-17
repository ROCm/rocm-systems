import { useMemo, useRef, useState } from 'react';

function defaultTargetSelection(targets) {
  const target = targets.includes('gfx1250') ? 'gfx1250' : targets[0];
  return target ? [target] : [];
}

function reconcileSelection(selected, options, fallback) {
  const available = selected.filter((value) => options.includes(value));
  return selected.length > 0 && available.length === 0 ? fallback : available;
}

export function useDashboardState(data) {
  const [targets, setTargets] = useState(() => defaultTargetSelection(data.targets));
  const [suites, setSuites] = useState(data.suites);
  const [historyRange, setHistoryRange] = useState('ALL');
  const [tab, setActiveTab] = useState('overview');
  const [benchmarkMode, setBenchmarkMode] = useState('single');
  const [search, setSearch] = useState('');
  const [comparisonBaselineId, setComparisonBaselineId] = useState(null);
  const [comparisonCandidateId, setComparisonCandidateId] = useState(null);
  const [explorerRunIds, setExplorerRunIds] = useState([]);

  const comparisonFilters = useRef(null);
  const setTab = (nextTab, { restoreComparisonFilters = true } = {}) => {
    if (tab === 'compare' && nextTab !== 'compare') {
      comparisonFilters.current = { targets, suites };
      setTargets(reconcileSelection(targets, data.targets, defaultTargetSelection(data.targets)));
      setSuites(reconcileSelection(suites, data.suites, data.suites));
    } else if (tab !== 'compare' && nextTab === 'compare' && restoreComparisonFilters) {
      if (comparisonFilters.current) {
        setTargets(comparisonFilters.current.targets);
        setSuites(comparisonFilters.current.suites);
      } else if (data.runs.length === 0) {
        setTargets(defaultTargetSelection(data.comparisonTargets));
        setSuites(data.comparisonSuites);
      }
    }
    setActiveTab(nextTab);
  };

  const filters = useMemo(() => ({ targets, suites }), [targets, suites]);

  return {
    filters,
    targets,
    suites,
    historyRange,
    tab,
    benchmarkMode,
    search,
    comparisonBaselineId,
    comparisonCandidateId,
    explorerRunIds,
    setTargets,
    setSuites,
    setHistoryRange,
    setTab,
    setBenchmarkMode,
    setSearch,
    setComparisonBaselineId,
    setComparisonCandidateId,
    setExplorerRunIds,
  };
}
