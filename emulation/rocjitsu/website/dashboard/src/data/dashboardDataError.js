export const MAX_DISPLAYED_VALIDATION_ERRORS = 10;

const validationFailureHeading = 'Dashboard data failed validation:';

export function summarizeDashboardDataError(error, limit = MAX_DISPLAYED_VALIDATION_ERRORS) {
  const message = error instanceof Error ? error.message : String(error);
  if (error?.dashboardDataErrorCode === 'unavailable') {
    const source = message.includes('raw.githubusercontent.com') ? ' on GitHub Raw' : '';
    return `Unable to reach published dashboard data${source}. Check your connection or retry.`;
  }
  if (error?.dashboardDataErrorCode === 'missing') {
    return 'No published test data is available.';
  }
  if (error?.dashboardDataErrorCode === 'invalid') {
    return `Published test data is invalid.\n${message}`;
  }
  if (!message.startsWith(`${validationFailureHeading}\n`)) return message;

  const details = message.slice(validationFailureHeading.length + 1).split('\n');
  if (details.length <= limit) return message;

  const omitted = details.length - limit;
  return [
    validationFailureHeading,
    ...details.slice(0, limit),
    `- … ${omitted} additional validation ${omitted === 1 ? 'error' : 'errors'} omitted`,
  ].join('\n');
}
