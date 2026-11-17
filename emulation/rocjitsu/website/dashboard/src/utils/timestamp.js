// The one definition of a usable published timestamp: a string the Date parser understands.
// Anything else (missing, a number, free text) is treated as absent, never as a date.
export const isValidTimestamp = (value) => typeof value === 'string' && Number.isFinite(Date.parse(value));
