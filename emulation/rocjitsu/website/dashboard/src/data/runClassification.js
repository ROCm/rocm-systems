export function isOfficialHistoryRun(run) {
  return run.trigger === 'auto' && run.branch === 'develop';
}
