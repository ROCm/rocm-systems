# isadiff-ci: rj_isa_diff on LLVM PRs, in a container

`isadiff-ci` runs `rj_isa_diff_pr` on every new head commit of a GitHub author's open
llvm/llvm-project PRs that change AMDGPU codegen. It also runs on local branches of an
llvm-project clone. Results land in a host directory. A draft PR comment is written for
every run, and `gh` posts it on request.

Everything runs in one Docker image. The host keeps a config, a token, the results and
the cron lines.

## Image

```sh
ci/isadiff-ci build REV     # isadiff-ci:<sha12> and isadiff-ci:latest
```

- **Stages.** The image is built from rocm-systems commit `REV`:
  - the tool stage builds `rj_isa_diff` from `git archive REV emulation/rocjitsu
    runtimes/api-headers` (without `ci/`);
  - the runtime stage has `rj_isa_diff`, `rj_isa_diff_pr` and `rj_isa_diff_mutants` in
    `/opt/isadiff/bin`, the scanner in `/opt/isadiff/ci`, clang-18 and lld-18 for the
    per-PR llc builds, ccache, git and gh.
- **Tool commit.** `ISADIFF_TOOL_SHA` records the tool commit; a tarball build has no git
  revision of its own.
- **Linker.** `RJ_ISA_DIFF_LLVM_BIN=/usr/lib/llvm-18/bin` provides `ld.lld` and
  `llvm-objcopy`. Each side assembles with the `llvm-mc` of its own llc build.
- **User.** The container runs as uid 1000 (`ci`), so files in the bind-mounted results
  directory belong to the host user with that uid.

## Setup (once)

```sh
mkdir -p ~/isadiff-ci/{bin,secrets,corpus,results,logs}
cp ci/isadiff-ci ~/isadiff-ci/bin/                 # a copy: later edits do not affect it
cp ci/config.example.json ~/isadiff-ci/config.json
# A classic token with only the public_repo scope (fine-grained tokens cannot comment on
# llvm/llvm-project):
install -m 600 /dev/stdin ~/isadiff-ci/secrets/gh_token <<< "$TOKEN"
cp -r SOME_TRITON_LLIR_DIR/. ~/isadiff-ci/corpus/  # optional: real kernels, deep scans
~/isadiff-ci/bin/isadiff-ci init                   # llvm-project clone into its volume
```

| Place | Holds |
|---|---|
| volume `isadiff-llvm` | the container's llvm-project clone (seeded from `~/llvm-project`, then fetched from GitHub) |
| volume `isadiff-builds` | the `--build-llc` dirs, one per PR (about 7.8 GB each) and one for local branches |
| volume `isadiff-ccache` | ccache, up to 60 GB |
| `~/isadiff-ci/results` | results, `index.md`, `events.log` |
| `~/llvm-project` | mounted read-only: seeds the clone and provides the local branches |

## Commands

```sh
isadiff-ci scan [--tier quick|deep]   # new heads of the configured PRs and local branches
isadiff-ci scan --pr N                # one PR, of any author
isadiff-ci post N [--yes]             # print (dry run) or post the latest run's comment
isadiff-ci reply N ID [--yes] < body  # reply in the thread of review comment ID
isadiff-ci gh ARGS...                 # gh with the token
isadiff-ci run CMD...                 # any command, e.g. rj_isa_diff replay ...
isadiff-ci shell
```

- **Tiers.**
  - `quick` runs `--build-llc` on the changed tests, with `--job-timeout 300`.
  - `deep` adds `--all-tests` and the corpus, with `--job-timeout 1800`.
  - A (head, tier) pair runs once. To run it again, delete its results directory.
- **Locking.** One scan runs at a time: a quick scan skips while another scan runs, and a
  deep scan waits up to two hours.
- **Cron lines.**

  ```
  */30 * * * * $HOME/isadiff-ci/bin/isadiff-ci scan --tier quick >> $HOME/isadiff-ci/logs/cron.log 2>&1
  10 4 * * * $HOME/isadiff-ci/bin/isadiff-ci scan --tier deep >> $HOME/isadiff-ci/logs/cron.log 2>&1
  ```

## Results

```
results/index.md                 latest run per subject and tier
results/events.log               first result and every status change
results/pr-N/<sha12>-<tier>/     run.json, driver.log, comment.md, posted.json,
                                 out/ (report.md, summary.json, functions/...)
results/branch-<name>/<sha12>-<tier>/
```

Triage of a `MISMATCH`, before calling it real:

1. Replay it: `isadiff-ci run rj_isa_diff replay --counterexample
   /work/results/.../case/counterexamples/rK-caseN.json --case /work/results/.../case`.
   Exit 0 means it reproduces.
2. Compare the counterexample's differing outputs with `diff before.s after.s`.
3. `poison-sensitive`: check whether the IR makes those inputs poison.
4. An `ASSUMED` or `UNRESOLVED` contract is not a verdict.

## Posting policy

- Nothing is published by default; `post N` prints the comment.
- `post N --yes` posts it, or edits the one earlier comment, found by the marker
  `<!-- rj-isa-diff -->`.
- With `"post": {"mode": "auto", "auto_prs": [N]}`, PASS and INCOMPLETE results of the
  listed PRs are posted after each scan. `MISMATCH`, `ERROR` and `TIMEOUT` never are.

## Limits

- **llc only.** Only llc output is compared, from the same IR on both sides. A change to
  opt passes alone has nothing to compare (`NO_APPLICABLE_TESTS`).
- **Not a proof.** A pass means no counterexample was found under the seed and budget.

## Tests

```sh
cd ci && python3 -m pytest -q test_ci.py                 # host, Python 3.10+
isadiff-ci run python3 -m pytest -q -p no:cacheprovider /opt/isadiff/ci/test_ci.py
```
