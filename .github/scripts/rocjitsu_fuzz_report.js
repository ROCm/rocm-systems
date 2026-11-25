// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

'use strict';

function parseNumber(environment, name, pattern, required) {
  const value = environment[name] || '';
  if (!pattern.test(value)) {
    if (required) {
      throw new Error(`Invalid ${name}: ${value}`);
    }
    return null;
  }
  return Number(value);
}

function parseCount(environment, name, required) {
  return parseNumber(environment, name, /^\d+$/, required);
}

function parseRate(environment, name, required) {
  return parseNumber(environment, name, /^\d+(?:\.\d+)?$/, required);
}

function parseOutcome(environment, name, required) {
  const value = environment[name] || '';
  if (['success', 'failure'].includes(value)) {
    return value;
  }
  if (required) {
    throw new Error(`Invalid ${name}: ${value}`);
  }
  return ['skipped', 'cancelled'].includes(value) ? value : 'unavailable';
}

function formatCount(value) {
  return value.toLocaleString('en-US');
}

function formatRate(value) {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function laneSummary(lane) {
  const details = [];
  if (lane.executionCount === null || lane.executionsPerSecond === null) {
    details.push('execution statistics unavailable');
  } else {
    details.push(
      `${formatCount(lane.executionCount)} input executions ` +
        `(${formatRate(lane.executionsPerSecond)} inputs/s)`,
    );
  }
  if (lane.crashCount !== null && lane.hangCount !== null) {
    details.push(
      `${lane.crashCount} crash input(s), ${lane.hangCount} hang input(s)`,
    );
  }
  return `${lane.outcome}; ${details.join('; ')}`;
}

function laneProgress(lane) {
  return (
    `${formatCount(lane.executionCount)} input executions ` +
    `(${formatRate(lane.executionsPerSecond)} inputs/s)`
  );
}

function buildReport(environment, { marker, phase, subject, runUrl }) {
  const completeJob = environment.FUZZ_JOB_RESULT === 'success';
  const lane = (prefix) => ({
    outcome: parseOutcome(environment, `${prefix}_CAMPAIGN_OUTCOME`, completeJob),
    crashCount: parseCount(environment, `${prefix}_CRASH_COUNT`, completeJob),
    executionCount: parseCount(
      environment,
      `${prefix}_EXECUTION_COUNT`,
      completeJob,
    ),
    executionsPerSecond: parseRate(
      environment,
      `${prefix}_EXECUTIONS_PER_SECOND`,
      completeJob,
    ),
    hangCount: parseCount(environment, `${prefix}_HANG_COUNT`, completeJob),
  });
  const asanUbsan = lane('ASAN_UBSAN');
  const tsan = lane('TSAN');
  const campaignSeconds = parseCount(
    environment,
    'CAMPAIGN_SECONDS',
    completeJob,
  );
  const workers = parseCount(environment, 'WORKERS', completeJob);
  const aggregateCrashCount = parseCount(
    environment,
    'CRASH_COUNT',
    completeJob,
  );
  const aggregateFindingCount = parseCount(
    environment,
    'FINDING_COUNT',
    completeJob,
  );
  const aggregateHangCount = parseCount(environment, 'HANG_COUNT', completeJob);
  const knownCrashCount = [asanUbsan.crashCount, tsan.crashCount].every(
    (value) => value !== null,
  )
    ? asanUbsan.crashCount + tsan.crashCount
    : null;
  const knownHangCount = [asanUbsan.hangCount, tsan.hangCount].every(
    (value) => value !== null,
  )
    ? asanUbsan.hangCount + tsan.hangCount
    : null;
  const crashCount = aggregateCrashCount ?? knownCrashCount;
  const hangCount = aggregateHangCount ?? knownHangCount;
  const findingCount =
    aggregateFindingCount ??
    (crashCount !== null && hangCount !== null ? crashCount + hangCount : 0);

  const campaignFailedValue = environment.CAMPAIGN_FAILED || '';
  if (completeJob && !['true', 'false'].includes(campaignFailedValue)) {
    throw new Error(`Invalid CAMPAIGN_FAILED: ${campaignFailedValue}`);
  }
  const campaignFailed = !completeJob || campaignFailedValue === 'true';
  if (
    completeJob &&
    (findingCount !== crashCount + hangCount ||
      crashCount !== knownCrashCount ||
      hangCount !== knownHangCount)
  ) {
    throw new Error('The fuzz finding counts are inconsistent.');
  }

  const artifactLine = environment.ARTIFACT_URL
    ? `[Download the partial outputs and AFL++ logs](${environment.ARTIFACT_URL}) (retained for 14 days).`
    : null;
  let body;
  if (campaignFailed) {
    body = [
      marker,
      `## rocjitsu ${phase} fuzz did not complete`,
      '',
      `The multiplexer campaigns for ${subject} ended before both sanitizer lanes completed.`,
      '',
      `- ASan+UBSan: ${laneSummary(asanUbsan)}`,
      `- TSan: ${laneSummary(tsan)}`,
      '',
      ...(findingCount === 0
        ? []
        : [
            '**A retained input can expose a pre-existing defect; this result does not by itself show that the PR introduced it.**',
            '',
          ]),
      ...(artifactLine ? [artifactLine, ''] : []),
      `[View the workflow run](${runUrl}).`,
    ].join('\n');
  } else {
    const durationMinutes = campaignSeconds / 60;
    const duration = Number.isInteger(durationMinutes)
      ? `${durationMinutes}-minute`
      : `${campaignSeconds}-second`;
    if (findingCount === 0) {
      body = [
        marker,
        `## rocjitsu ${phase} fuzz passed`,
        '',
        `The ${workers}-worker multiplexer campaigns for ${subject} completed without recording a crash or hang.`,
        '',
        `- ASan+UBSan (${duration}): ${laneProgress(asanUbsan)}`,
        `- TSan (${duration}): ${laneProgress(tsan)}`,
        '',
        `[View the workflow run](${runUrl}).`,
      ].join('\n');
    } else {
      body = [
        marker,
        `## rocjitsu ${phase} fuzz finding`,
        '',
        `The ${workers}-worker multiplexer campaigns for ${subject} recorded **${crashCount} crash input(s)** and **${hangCount} hang input(s)**.`,
        '',
        `- ASan+UBSan (${duration}): ${laneProgress(asanUbsan)}; ${asanUbsan.crashCount} crash input(s), ${asanUbsan.hangCount} hang input(s)`,
        `- TSan (${duration}): ${laneProgress(tsan)}; ${tsan.crashCount} crash input(s), ${tsan.hangCount} hang input(s)`,
        '',
        '**A retained input can expose a pre-existing defect; this result does not by itself show that the PR introduced it.**',
        '',
        `[Download the saved inputs and AFL++ statistics](${environment.ARTIFACT_URL}) (retained for 14 days).`,
        '',
        `[View the workflow run](${runUrl}).`,
      ].join('\n');
    }
  }
  return { body, campaignFailed, findingCount };
}

function findExistingReportComment(comments, marker) {
  return comments.find(
    (comment) =>
      comment.user?.login === 'github-actions[bot]' &&
      comment.performed_via_github_app?.slug === 'github-actions' &&
      comment.body?.includes(marker),
  );
}

module.exports = { buildReport, findExistingReportComment };
