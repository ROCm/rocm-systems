export const dashboardPages = [
  { id: 'overview', label: 'Overview' },
  { id: 'branch', label: 'Branch runs' },
  { id: 'benchmarks', label: 'Benchmarks' },
  { id: 'compare', label: 'Run comparison' },
];

export const dashboardPageLabels = Object.fromEntries(dashboardPages.map(({ id, label }) => [id, label]));
