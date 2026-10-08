import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from 'vitest';

const workflow = readFileSync(new URL('../../../../../../.github/workflows/rocjitsu-benchmarks.yml', import.meta.url), 'utf8');
const policyStep = workflow.match(/ {6}- name: Check manual publication corpus\n([\s\S]*?)(?=\n {6}- name:)/)[1];
const script = policyStep.split('        run: |\n')[1]
  .split('\n').map((line) => line.replace(/^ {10}/, '')).join('\n');
const artifactRoot = fileURLToPath(new URL('../../test-results/', import.meta.url));
const officialSha = 'a'.repeat(40);

function checkPolicy(officialWorkflow, corpusSha) {
  mkdirSync(artifactRoot, { recursive: true });
  const workspace = mkdtempSync(path.join(artifactRoot, 'publication-policy-'));
  try {
    const workflows = path.join(workspace, 'dashboard/.github/workflows');
    mkdirSync(workflows, { recursive: true });
    writeFileSync(path.join(workflows, 'rocjitsu-benchmarks.yml'), officialWorkflow);
    return spawnSync('bash', ['-e', '-o', 'pipefail', '-c', script], {
      env: { ...process.env, GITHUB_WORKSPACE: workspace, CORPUS_SHA: corpusSha },
      encoding: 'utf8',
    });
  } finally {
    rmSync(workspace, { recursive: true, force: true });
  }
}

test('checks manual pins against the canonical develop checkout before publishing', () => {
  const checkout = workflow.match(/ {6}- name: Checkout dashboard validator\n([\s\S]*?)(?=\n {6}- name:)/)[1];
  expect(checkout).toContain('repository: ROCm/rocm-systems');
  expect(checkout).toContain('ref: develop');
  expect(checkout).toContain('.github/workflows');
  expect(policyStep).toContain("if: github.event_name == 'workflow_dispatch'");
  expect(workflow.indexOf('name: Check manual publication corpus')).toBeLessThan(workflow.indexOf('name: Publish dashboard-ready result'));
});

test.each(['', ' # companion PR'])('accepts the official corpus pin%s', (annotation) => {
  const result = checkPolicy(`env:\n  CORPUS_SHA: ${officialSha}${annotation}\n`, officialSha);
  expect(result.status, result.stderr).toBe(0);
});

test('rejects an experimental corpus before its test definitions can be published', () => {
  const result = checkPolicy(`env:\n  CORPUS_SHA: ${officialSha}\n`, 'b'.repeat(40));
  expect(result.status).toBe(1);
  expect(result.stdout).toContain("Manual publication requires develop's corpus pin");
  expect(result.stdout).toContain('diagnostics remain available as artifacts');
});

test.each([
  '',
  'env:\n  CORPUS_SHA: not-a-sha\n',
  `env:\n  CORPUS_SHA: ${officialSha}\n  CORPUS_SHA: ${'b'.repeat(40)}\n`,
])('rejects missing, invalid, or ambiguous official pin: %s', (officialWorkflow) => {
  const result = checkPolicy(officialWorkflow, officialSha);
  expect(result.status).toBe(1);
  expect(result.stdout).toContain('Cannot read the official corpus pin');
});
