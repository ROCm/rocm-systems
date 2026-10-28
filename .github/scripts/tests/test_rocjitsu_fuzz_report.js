// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
  buildReport,
  findExistingReportComment,
} = require('../rocjitsu_fuzz_report.js');

const context = {
  marker: '<!-- marker -->',
  phase: 'presubmit',
  subject: 'a test commit',
  runUrl: 'https://example.test/run',
};

function completeEnvironment(overrides = {}) {
  return {
    ARTIFACT_URL: 'https://example.test/artifact',
    ASAN_UBSAN_CAMPAIGN_OUTCOME: 'success',
    ASAN_UBSAN_CRASH_COUNT: '0',
    ASAN_UBSAN_EXECUTION_COUNT: '1234',
    ASAN_UBSAN_EXECUTIONS_PER_SECOND: '12.5',
    ASAN_UBSAN_HANG_COUNT: '0',
    TSAN_CAMPAIGN_OUTCOME: 'success',
    TSAN_CRASH_COUNT: '0',
    TSAN_EXECUTION_COUNT: '5678',
    TSAN_EXECUTIONS_PER_SECOND: '6.25',
    TSAN_HANG_COUNT: '0',
    CRASH_COUNT: '0',
    FINDING_COUNT: '0',
    HANG_COUNT: '0',
    CAMPAIGN_FAILED: 'false',
    CAMPAIGN_SECONDS: '360',
    FUZZ_JOB_RESULT: 'success',
    WORKERS: '32',
    ...overrides,
  };
}

test('reports a successful campaign with progress', () => {
  const report = buildReport(completeEnvironment(), context);

  assert.equal(report.campaignFailed, false);
  assert.equal(report.findingCount, 0);
  assert.match(report.body, /rocjitsu presubmit fuzz passed/);
  assert.match(report.body, /1,234 input executions \(12\.50 inputs\/s\)/);
  assert.match(report.body, /5,678 input executions \(6\.25 inputs\/s\)/);
});

test('reports a build failure without step outputs', () => {
  const report = buildReport(
    {
      FUZZ_JOB_RESULT: 'failure',
      CAMPAIGN_SECONDS: '360',
      WORKERS: '32',
    },
    context,
  );

  assert.equal(report.campaignFailed, true);
  assert.match(report.body, /rocjitsu presubmit fuzz did not complete/);
  assert.match(report.body, /ASan\+UBSan: unavailable; execution statistics unavailable/);
  assert.match(report.body, /TSan: unavailable; execution statistics unavailable/);
  assert.doesNotMatch(report.body, /Download/);
});

test('reports skipped sanitizer lanes from a build failure', () => {
  const report = buildReport(
    completeEnvironment({
      ARTIFACT_URL: 'https://example.test/build-failure-artifact',
      ASAN_UBSAN_CAMPAIGN_OUTCOME: 'skipped',
      ASAN_UBSAN_EXECUTION_COUNT: '0',
      TSAN_CAMPAIGN_OUTCOME: 'skipped',
      TSAN_EXECUTION_COUNT: '0',
      CAMPAIGN_FAILED: 'true',
      FUZZ_JOB_RESULT: 'failure',
    }),
    context,
  );

  assert.equal(report.campaignFailed, true);
  assert.equal(report.findingCount, 0);
  assert.match(report.body, /rocjitsu presubmit fuzz did not complete/);
  assert.match(report.body, /ASan\+UBSan: skipped; 0 input executions/);
  assert.match(report.body, /TSan: skipped; 0 input executions/);
  assert.match(report.body, /Download the partial outputs and AFL\+\+ logs/);
});

test('preserves findings when artifact upload fails', () => {
  const report = buildReport(
    completeEnvironment({
      ARTIFACT_URL: '',
      ASAN_UBSAN_CAMPAIGN_OUTCOME: 'failure',
      ASAN_UBSAN_CRASH_COUNT: '2',
      CRASH_COUNT: '2',
      FINDING_COUNT: '2',
      CAMPAIGN_FAILED: 'true',
      FUZZ_JOB_RESULT: 'failure',
    }),
    context,
  );

  assert.equal(report.findingCount, 2);
  assert.match(report.body, /2 crash input\(s\), 0 hang input\(s\)/);
  assert.match(report.body, /pre-existing defect/);
  assert.doesNotMatch(report.body, /Download/);
});

test('updates only a marker owned by the GitHub Actions app', () => {
  const human = {
    id: 1,
    user: { login: 'reviewer' },
    body: context.marker,
  };
  const bot = {
    id: 2,
    user: { login: 'github-actions[bot]' },
    performed_via_github_app: { slug: 'github-actions' },
    body: context.marker,
  };

  assert.equal(findExistingReportComment([human, bot], context.marker), bot);
  assert.equal(findExistingReportComment([human], context.marker), undefined);
});
