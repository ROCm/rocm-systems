# Commit equivalence checklist

Use multiple signals. No single command proves semantic equivalence in all
cases.

## Evidence hierarchy

1. **Reachability:** the exact topic commit is an ancestor of new upstream.
2. **Stable patch identity:** normalized patch IDs match across histories.
3. **Range comparison:** `git cherry` and `git range-diff` associate a topic
   patch with an upstream patch.
4. **Tree evidence:** every material topic hunk or behavior exists upstream,
   possibly after file movement or refactoring.
5. **Behavior evidence:** focused tests demonstrate the same contract and edge
   cases.
6. **Message/PR metadata:** useful for locating candidates, never sufficient by
   itself.

## Per-commit questions

- Is the exact OID reachable from upstream?
- Does a stable patch ID match an upstream commit?
- Was the change squash-merged with sibling commits?
- Was it cherry-picked with conflict resolution or edited before merge?
- Did upstream independently implement only part of it?
- Were files renamed, generated, or broadly reformatted?
- Does the topic commit contain tests, documentation, or build changes that the
  upstream implementation omitted?
- Do later topic commits depend on its intermediate tree even if its final
  behavior is upstream?
- Would dropping it change the final diff against new upstream?

## Generated changes

Treat generator inputs and output as one logical change. A generated patch may
have a different patch ID because the generator or source specification moved.
Compare generator intent, current regeneration result, affected ISA coverage,
and tests. Never retain stale generated output merely to preserve a commit.

## Classification record

For each old commit record:

| Old commit | Classification | Upstream evidence | Planned action | Dependency/risk |
| --- | --- | --- | --- | --- |
| `<oid> subject` | ancestor / patch-equivalent / absorbed / partial / topic-only / uncertain | OID, patch ID, files, tests | drop / replay / edit / split | concise note |

After rewriting, add the corresponding new commit or upstream provider so every
old commit is accounted for.
