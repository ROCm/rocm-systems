# ROCm Libraries Gardeners

This documents the mechanics of
[gardening](https://github.com/ROCm/TheRock/blob/main/docs/rfcs/RFC0002-MonoRepo-Gardener-Rotations.md)
for the ROCm Libraries. If you haven't read the above doc, please start there.

## Becoming a member

Gardeners will need to be members of the [Compute Library Gardeners team](https://github.com/orgs/ROCm/teams/systems-gardeners).
Please contact an owner to become a gardener.

## Communications channel

We will be leveraging a shared Teams channel that contains all gardeners as well as core
infrastructure team members. You will be added to this channel once you become a member.

For anyone who wants to reach a gardener please email:
[rocm-systems-gardeners](mailto:rocm-systems-gardeners@amd.com)

## Mechanics of Gardening

Your primary job is to keep the mono-repo shippable. In order to facilitate this we've made
status badges for all relevant CI available here:
https://github.com/ROCm/rocm-systems?tab=readme-ov-file#monorepo-status-and-ci-health.
Effectively your job is to ensure all status badges are green. All of these status
badges are clickable which will allow you to deep-dive on any failures quickly. If any
CI is missing, please file an issue leveraging the "gardener" tag, ping on the teams chat,
or preferably, add it yourself. You'll probably be tagged to review the PR if someone
else gets to it first.

## Notes on Privileges

Developers will not be able to bypass pre-submit checks in this repository unless an admin or
gardener pushes it through. This is being done intentionally to ensure we keep the quality of
the tree green. This also means that you will be asked to push changes through without
additional context. Your duty is to ensure you keep the tree green (or make it greener) so gardeners will need to understand the context before approving
any of these changes. Changes
that are ok:

- Reverts to fix broken things.
- Fast-forward fixes where reverts are unclear
- Fixes unrelated to code health (docs, etc)

On a case by case basis you should consider critical customer fixes, but these should be considered
as a group and likely admins should be approving the majority of those.

As an example to include an admin: *we have a critical feature but develop is broken and it is unrelated to our changes*

## Rotation

Week | North America | Europe / India / APAC
---- | ------- | ---------


It is the responsibility of the current gardeners to update the table when the gardeners rotate.

### Log

Filling in this section is optional while on rotation. While this level of
organization and tracking is not expected from all members, seeing the incident
history and actions taken in one location can be useful. However, for bugs that you can't immediately address
please file a new GH issue and label it with the "gardener" label.

You can see current list of [gardener known bugs](https://github.com/ROCm/rocm-systems/issues?q=is%3Aissue%20state%3Aopen%20label%3Agardener)

Date | Library | Issue overview | Link to details | Resolved?
---- | ------- | -------------- | --------------- | ---------
