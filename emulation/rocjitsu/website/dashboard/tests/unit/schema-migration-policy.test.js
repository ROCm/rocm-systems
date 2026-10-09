import { readFileSync } from 'node:fs';
import { expect, test } from 'vitest';

test('migration documentation records the accepted instrumented-run exclusion policy and its limits', () => {
  const document = readFileSync(new URL('../../docs/schema-migration.md', import.meta.url), 'utf8');
  const policy = document.split('## Accepted exclusion policy')[1]?.split('\n## ')[0] ?? '';
  expect(policy).toContain('Non-Vanilla instrumented runs are excluded from dashboard history and comparisons.');
  expect(policy).toContain('Original records are retained unchanged in raw exports');
  expect(policy).toContain('never relabelled as Vanilla');
  expect(policy).toContain('does not restore reserved plugin support or retired UI');
});
