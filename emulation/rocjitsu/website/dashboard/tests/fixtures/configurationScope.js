// Test-only selected-matrix editing. The benchmark-set lookup remains stable.
export function setFixtureConfigurations(_publication, run, rows) {
  run.benchmark_results = rows;
}
