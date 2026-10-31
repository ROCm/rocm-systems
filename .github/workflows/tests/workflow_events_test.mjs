// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { Lexer, Parser, Evaluator, data } from '@actions/expressions';
import { parse } from 'yaml';

function context(value) {
  if (value === null) return new data.Null();
  if (typeof value === 'string') return new data.StringData(value);
  if (typeof value === 'number') return new data.NumberData(value);
  if (typeof value === 'boolean') return new data.BooleanData(value);
  return new data.Dictionary(
    ...Object.entries(value).map(([key, value]) => ({
      key,
      value: context(value),
    })),
  );
}

// Evaluate the workflow's actual expressions with GitHub's expression library.
function evaluate(expression, github, matrix = {}) {
  const tokens = new Lexer(expression).lex().tokens;
  const parsed = new Parser(tokens, ['github', 'matrix'], []).parse();
  return new Evaluator(parsed, context({ github, matrix })).evaluate();
}

function render(template, github, matrix) {
  return template.replace(/\$\{\{(.*?)\}\}/g, (_, expression) =>
    evaluate(expression, github, matrix).coerceString(),
  );
}

for (const [filename, jobName] of [
  ['scoped-formatting.yml', 'pre-commit'],
  ['unit-tests.yml', 'test'],
]) {
  const workflow = parse(
    readFileSync(new URL(`../${filename}`, import.meta.url), 'utf8'),
  );
  const job = workflow.jobs[jobName];
  const github = {
    workflow: workflow.name,
    sha: 'head',
    run_id: 1,
    event: { number: 42, action: 'synchronize' },
  };
  const validationGroup = render(workflow.concurrency.group, github);
  const requiredNames = job.strategy.matrix.include.map(matrix =>
    render(job.name, github, matrix),
  );

  test(`${filename}: source events and retargets run validation`, () => {
    // PR events must reach scope selection without branch or path filtering.
    for (const filter of ['branches', 'branches-ignore', 'paths', 'paths-ignore']) {
      assert.equal(workflow.on.pull_request[filter], undefined);
    }
    for (const action of ['opened', 'synchronize', 'reopened', 'ready_for_review']) {
      assert(workflow.on.pull_request.types.includes(action));
      const event = { ...github, event: { number: 42, action } };
      assert.equal(evaluate(`!!(${job.if})`, event).coerceString(), 'true');
      assert.equal(render(workflow.concurrency.group, event), validationGroup);
    }
    assert(workflow.on.pull_request.types.includes('edited'));
    const retarget = {
      ...github,
      event: {
        number: 42,
        action: 'edited',
        changes: { base: { ref: { from: 'release' } } },
      },
    };
    assert.equal(evaluate(`!!(${job.if})`, retarget).coerceString(), 'true');
    assert.equal(render(workflow.concurrency.group, retarget), validationGroup);
    assert.deepEqual(
      job.strategy.matrix.include.map(matrix => render(job.name, retarget, matrix)),
      requiredNames,
    );
    assert(workflow.on.push.branches.includes('develop'));
    const push = { ...github, event: { before: 'base' } };
    assert.equal(evaluate(`!!(${job.if})`, push).coerceString(), 'true');
  });

  test(`${filename}: metadata edits neither cancel nor replace validation`, () => {
    for (const changes of [{ title: { from: 'old' } }, { body: { from: 'old' } }, {}]) {
      const metadata = {
        ...github,
        run_id: 2,
        event: { number: 42, action: 'edited', changes },
      };
      assert.equal(evaluate(`!!(${job.if})`, metadata).coerceString(), 'false');
      assert.notEqual(render(workflow.concurrency.group, metadata), validationGroup);
      // Job conditions are evaluated before matrix expansion. The skipped
      // job must also be nonrequired with no matrix context at all.
      assert(!requiredNames.includes(render(job.name, metadata, {})));
      for (const matrix of job.strategy.matrix.include) {
        assert(!requiredNames.includes(render(job.name, metadata, matrix)));
      }
    }
  });
}
