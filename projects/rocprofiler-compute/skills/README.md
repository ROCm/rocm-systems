# rocprofiler-compute agent skills

User-facing Agent Skills that teach an AI agent to drive `rocprof-compute`.
Each skill covers one profiling capability end to end, from collecting the data
to interpreting it, so an agent loads only what the question needs.

| Skill | Use it for |
|---|---|
| [kernel-bottleneck](kernel-bottleneck/SKILL.md) | Start here. Profile an application, find the hot kernel, check occupancy and scheduler limits |
| [speed-of-light](speed-of-light/SKILL.md) | How close a kernel runs to hardware peak |
| [memory](memory/SKILL.md) | Memory chart, caches, LDS, fabric traffic, gfx950 bandwidth analysis |
| [roofline](roofline/SKILL.md) | Compute-bound versus memory-bound, arithmetic intensity |
| [pc-sampling](pc-sampling/SKILL.md) | Which instruction is hot and why it stalls |
| [torch-trace](torch-trace/SKILL.md) | Attribute GPU kernels to PyTorch and Triton operators |

These are user-facing skills. The contributor workflows under `.ai/skills/`
(code review, rebase) are a separate thing and are not published here.

## Structure

Every skill directory holds:

```
<skill-name>/
├── SKILL.md            # frontmatter (name, description) plus the instructions
├── skill-card.md       # human-facing card: description, owner, license
└── evals/evals.json    # prompts that must and must not route to this skill
```

`SKILL.md` states the decision: which command to run, when, and where the
boundaries are. Explanations stay in `docs/`, linked from the skill, so the
same fact is not maintained in two places.

## Tests

There is no CI for skills. `run_evals.py` is the gate, and it is meant to be
run by hand before a release:

```bash
./skills/run_evals.py --list                 # validate the datasets, run nothing
./skills/run_evals.py                        # routing: does the right skill load?
./skills/run_evals.py --mode behavioral      # also grade what the agent did
./skills/run_evals.py --skill memory         # one skill
```

Routing mode denies the agent its tools and only checks which skill it chose,
so it is quick and needs no GPU. Behavioral mode lets the agent work and grades
`logs_contain`, `files_exist`, and the plain-language `expected_behavior` and
`unexpected_behavior` claims, so it needs a supported GPU and real time.

Both modes drive the `claude` CLI. The script exits 77 when that CLI is absent,
so a caller can treat it as skipped rather than failed.

Each dataset needs at least three cases that should trigger the skill, two that
should not, and one triggering case with judged behavior. Negative cases carry
only an id, a prompt, and an optional note.

## Keeping skills correct

There is no CI to catch drift, so it falls to each change. A PR that alters CLI
options, output layout, supported architectures, or the experimental status of
a feature updates the matching skill in the same PR, exactly as it updates docs
and tests.
