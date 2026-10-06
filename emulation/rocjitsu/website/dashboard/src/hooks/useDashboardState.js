import { useMemo, useState } from 'react';
import { selectThreadingModeData } from '../data/dashboardValidation';

function defaultTargets(data) {
  const target = data.targets.includes('gfx1250') ? 'gfx1250' : data.targets[0];
  return target ? [target] : [];
}

export function reconcileSelection(selected, available, fallback) {
  const retained = selected.filter((value) => available.includes(value));
  return selected.length > 0 && retained.length === 0 ? fallback : retained;
}

export function useDashboardState(allData) {
  const availableThreadingModes = useMemo(() => ['default', 'single'].filter((mode) => (
    [...allData.runs, ...allData.pluginRuns].some((run) => run.threadingMode === mode)
  )), [allData]);
  const [threadingModes, updateThreadingModes] = useState(() => availableThreadingModes.slice(0, 1));
  const data = useMemo(() => selectThreadingModeData(allData, threadingModes), [allData, threadingModes]);
  const [targets, setTargets] = useState(() => defaultTargets(data));
  const [suites, setSuites] = useState(data.suites);
  const [historyRange, setHistoryRange] = useState('ALL');
  const [tab, setTab] = useState('overview');
  const [benchmarkMode, setBenchmarkMode] = useState('single');
  const [search, setSearch] = useState('');
  const [comparisonBaselineId, setComparisonBaselineId] = useState(null);
  const [comparisonCandidateId, setComparisonCandidateId] = useState(null);
  const [explorerRunIds, setExplorerRunIds] = useState([]);

  const setThreadingModes = (modes) => {
    const selected = availableThreadingModes.filter((mode) => modes.includes(mode));
    const nextData = selectThreadingModeData(allData, selected);
    if (selected.length > 0) {
      setTargets((current) => reconcileSelection(current, nextData.targets, defaultTargets(nextData)));
      setSuites((current) => reconcileSelection(current, nextData.suites, nextData.suites));
    }
    setComparisonBaselineId(null);
    setComparisonCandidateId(null);
    setExplorerRunIds([]);
    updateThreadingModes(selected);
  };

  const filters = useMemo(() => ({ targets, suites }), [targets, suites]);

  return {
    data,
    threadingModes,
    availableThreadingModes,
    setThreadingModes,
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
