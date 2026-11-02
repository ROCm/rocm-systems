export const dashboardPages = [
  { id: 'overview', label: 'Overview' },
  { id: 'branch', label: 'Branch Runs' },
  { id: 'benchmarks', label: 'Benchmarks' },
  { id: 'compare', label: 'Run Comparison' },
];

export const dashboardPageLabels = Object.fromEntries(dashboardPages.map(({ id, label }) => [id, label]));
