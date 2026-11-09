# rocprofiler-compute agent skills

User-facing Agent Skills that teach an AI agent to drive `rocprof-compute`.
Each skill covers one profiling capability end to end, from collecting the data
to interpreting it, so an agent loads only what the question needs.

| Skill | Use it for |
|---|---|
| [rocprof-compute-kernel-bottleneck](rocprof-compute-kernel-bottleneck/SKILL.md) | Start here. Profile an application, find the hot kernel, check occupancy and scheduler limits |
| [rocprof-compute-speed-of-light](rocprof-compute-speed-of-light/SKILL.md) | How close a kernel runs to hardware peak |
| [rocprof-compute-memory](rocprof-compute-memory/SKILL.md) | Memory chart, caches, LDS, fabric traffic, gfx950 bandwidth analysis |
| [rocprof-compute-roofline](rocprof-compute-roofline/SKILL.md) | Compute-bound versus memory-bound, arithmetic intensity |
| [rocprof-compute-pc-sampling](rocprof-compute-pc-sampling/SKILL.md) | Which instruction is hot and why it stalls |
| [rocprof-compute-torch-trace](rocprof-compute-torch-trace/SKILL.md) | Attribute GPU kernels to PyTorch and Triton operators |

These are user-facing skills. The contributor workflows under `.ai/skills/`
(code review, rebase) are a separate thing and are not published here.

## Install

A ROCm install ships these skills under
`<rocm-path>/share/rocprofiler-compute/skills/`, matching the installed
`rocprof-compute`. `rocprof-compute --help` prints that path.
Run the install script from there for your agent:

```bash
<rocm-path>/share/rocprofiler-compute/skills/install-skills.sh --agent claude
```

| `--agent` | Installs into |
|---|---|
| `claude` | `~/.claude/skills/` |
| `codex` | `~/.agents/skills/` |
| `cursor` | `~/.cursor/skills/` |

From a source checkout, run `skills/install-skills.sh` the same way.

The script replaces any `rocprof-compute-*` skills already in that folder, so
run it again after upgrading ROCm. Cursor also reads the Claude and Codex
folders, so install for one agent only if you use Cursor alongside them.

## Uninstall

```bash
<rocm-path>/share/rocprofiler-compute/skills/install-skills.sh --agent claude --uninstall
```

## Developing skills

The rest of this file is for people who edit these skills. The skills are also
published in [amd/skills](https://github.com/amd/skills), the AMD skills
catalog, so a change here must work in this repo and in that catalog. See
[Publishing to amd/skills](#publishing-to-amdskills).

### Structure

Every skill directory holds:

```
<skill-name>/
├── SKILL.md            # frontmatter (name, description) plus the instructions
├── skill-card.md       # human-facing card: description, owner, license
└── evals/
    ├── evals.json            # cases this repo and amd/skills run
    ├── extended_evals.json   # optional; cases only this repo runs
    ├── machine.yml           # hardware amd/skills runs evals.json on
    └── samples/              # copies of the samples the cases use
```

`SKILL.md` states the decision: which command to run, when, and where the
boundaries are. Explanations stay in `docs/`, linked from the skill, so the
same fact is not maintained in two places. Link docs as `../../docs/...`:
`install-skills.sh` rewrites those links to the docs installed next to the
skills, so they keep working offline.

Both dataset files use the
[skillscope format](https://github.com/amd/skillscope/blob/main/docs/authoring-evals.md).
A case that needs an application sets `workspace` to `evals/samples`, and its
prompt names the application. The contents of that directory are copied to the
workspace root when the case runs. Each copy in `evals/samples/` and its
original under `sample/` name each other in a header comment; change them
together.

### Tests

CI runs the [skillscope](https://github.com/amd/skillscope) structural checks,
the same checks amd/skills runs, on every pull request that changes the skills
or `docs/`. They cover both dataset files. CI does not grade routing or
behavior: routing needs a model API key, and behavior needs a supported GPU.

`run_evals.py` grades both and is the release gate. It runs `evals.json` and
`extended_evals.json` together. `evals.json` targets Strix Halo, except for
pc-sampling, and most extended cases need an Instinct GPU, so pick the cases
for your GPU with `--skill` or `--only`:

```bash
./skills/run_evals.py --list                         # validate the datasets
./skills/run_evals.py                                # does the right skill load?
./skills/run_evals.py --mode behavioral              # also grade the agent
./skills/run_evals.py --skill rocprof-compute-memory  # one skill
./skills/run_evals.py --only mem-cache-hit-rates      # one case
```

Routing mode denies the agent its tools and only checks which skill it chose,
so it is quick and needs no GPU. Behavioral mode lets the agent work and grades
`logs_contain`, `files_exist`, and the plain-language `expected_behavior` and
`unexpected_behavior` claims, so it needs a supported GPU and real time. It
also fails a case when any `rocprof-compute` command the agent ran failed. The
agent runs commands without asking for approval, so use a test machine.

Both modes install the skills with `install-skills.sh`, so they also test the
install script and its docs links. Both drive the `claude` CLI. The script
exits 77 when that CLI is absent, so a caller can treat it as skipped rather
than failed.

### Keeping skills correct

CI checks only structure and links, so keeping the content correct falls to
each change. A PR that alters CLI options, output layout, supported
architectures, or the experimental status of a feature updates the matching
skill in the same PR, exactly as it updates docs and tests.

Two sections are shared word for word. "Before you start" is in all six skills.
"Collect only what the question needs" is in the five that collect counters:
kernel-bottleneck, memory, roofline, speed-of-light, and torch-trace. Keep each
shared section identical wherever it appears. If a tip applies to only one
skill, put it in that skill's own steps.

Skills must also serve users on the two previous ROCm releases. When a change
alters how a feature behaves, keep a short note in the skill saying how it
worked before and in which release.

## Publishing to amd/skills

amd/skills imports each skill directory from `develop`. It copies only the skill
directory and rewrites links that leave it, such as `../../docs/...`, to GitHub
URLs. Edit the skills here; the next import overwrites edits made there.

amd/skills grades each skill with skillscope under these rules:

- It runs only `evals/evals.json`. That file needs at least three cases that
  should trigger the skill, two that should not, and one triggering case with
  `expected_behavior` or `unexpected_behavior`. `run_evals.py` checks the same
  minimum.
- It runs `evals.json` on Strix Halo (gfx1151) unless `evals/machine.yml` asks
  for other hardware. Every skill sets `os: [Linux]` there to skip the Windows
  runners. Only pc-sampling also sets `labels`, because it needs an MI300X; that
  sends it to the catalog's gated Instinct runners.
- Put cases that cannot pass on those runners in `evals/extended_evals.json`:
  cases that need an Instinct GPU, and the torch-trace cases that profile a
  model, since the catalog runners may not have PyTorch. The catalog skips this
  file.
- It copies only the skill directory, so `evals/samples/` holds copies of the
  samples from `sample/`.
