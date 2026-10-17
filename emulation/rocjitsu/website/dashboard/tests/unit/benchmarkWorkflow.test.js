import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from 'vitest';

const workflow = readFileSync(new URL('../../../../../../.github/workflows/rocjitsu-benchmarks.yml', import.meta.url), 'utf8');
const policyStep = workflow.match(/ {6}- name: Check publication workflow\n([\s\S]*?)(?=\n {6}- name:)/)[1];
const script = policyStep.split('        run: |\n')[1]
  .split('\n').map((line) => line.replace(/^ {10}/, '')).join('\n');
const artifactRoot = fileURLToPath(new URL('../../test-results/', import.meta.url));

function checkPolicy(officialWorkflow, runWorkflow) {
  mkdirSync(artifactRoot, { recursive: true });
  const workspace = mkdtempSync(path.join(artifactRoot, 'publication-policy-'));
  try {
    for (const [checkout, content] of [['dashboard', officialWorkflow], ['publication', runWorkflow]]) {
      if (content === null) continue;
      const workflows = path.join(workspace, checkout, '.github/workflows');
      mkdirSync(workflows, { recursive: true });
      writeFileSync(path.join(workflows, 'rocjitsu-benchmarks.yml'), content);
    }
    return spawnSync('bash', ['-e', '-o', 'pipefail', '-c', script], {
      env: { ...process.env, GITHUB_WORKSPACE: workspace },
      encoding: 'utf8',
    });
  } finally {
    rmSync(workspace, { recursive: true, force: true });
  }
}

test('checks the executed workflow against canonical develop before publishing for every trigger', () => {
  const checkout = workflow.match(/ {6}- name: Checkout dashboard validator\n([\s\S]*?)(?=\n {6}- name:)/)[1];
  const publicationCheckout = workflow.match(/ {6}- name: Checkout repository for publication\n([\s\S]*?)(?=\n {6}- name:)/)[1];
  expect(checkout).toContain('repository: ROCm/rocm-systems');
  expect(checkout).toContain('ref: develop');
  expect(checkout).toContain('.github/workflows');
  expect(publicationCheckout).toContain('ref: ${{ github.sha }}');
  expect(policyStep).not.toContain('if:');
  expect(workflow.indexOf('name: Check publication workflow')).toBeLessThan(workflow.indexOf('name: Publish dashboard-ready result'));
});

test('accepts identical workflow contents', () => {
  const result = checkPolicy(workflow, workflow);
  expect(result.status, result.stderr).toBe(0);
});

test.each([
  ['corpus pin', (contents) => contents.replace(/CORPUS_SHA: [0-9a-f]{40}/, `CORPUS_SHA: ${'b'.repeat(40)}`)],
  ['publisher invocation', (contents) => contents.replace('--is-beta', '--different-option')],
  ['comment', (contents) => `${contents}\n# Branch-only change\n`],
])('rejects a changed %s even when the workflow remains valid YAML', (_label, change) => {
  const result = checkPolicy(workflow, change(workflow));
  expect(result.status).toBe(1);
  expect(result.stdout).toContain('Publication requires the benchmark workflow to match canonical develop');
  expect(result.stdout).toContain('diagnostics remain available as artifacts');
});

test('rejects an outdated workflow when canonical develop has advanced', () => {
  const result = checkPolicy(workflow.replace('--is-beta', '--new-publisher-option'), workflow);
  expect(result.status).toBe(1);
});

test.each([
  ['official', null, workflow],
  ['executed', workflow, null],
])('rejects a missing %s workflow', (_label, officialWorkflow, runWorkflow) => {
  const result = checkPolicy(officialWorkflow, runWorkflow);
  expect(result.status).toBe(1);
});
