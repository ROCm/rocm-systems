# Rocjitsu web applications

This directory contains the public web applications for Rocjitsu.

| Directory | Purpose | Published path |
| --- | --- | --- |
| [dashboard/](dashboard/README.md) | Simulation-performance dashboard (React + Vite) | `/rocjitsu-dashboard/` |
| [handbook/](handbook/README.md) | Markdown documentation and blog (MkDocs) | `/rocjitsu/` |

## Dashboard data input

Workflow authors should start with the dashboard
[data-generation guide](dashboard/docs/data-generation-guide.md) and follow the
[authoritative schema-2 contract](dashboard/docs/website-data-contract.md).
The [dashboard docs index](dashboard/docs/README.md) also links build and test guidance.

Each application owns its dependencies, build configuration, and tests. See its
README for local development instructions. The handbook reads its content
directly from [../docs/](../docs/), with [../README.md](../README.md) supplying
the homepage. These are the maintained sources; the website has no content copy.

The [publishing workflow](../../../.github/workflows/rocjitsu-publish-website.yml)
compares current `develop` with each site's last publication and builds only affected
sites to their separate directories on `gh-pages`. Manual runs and changes to
the shared publishing workflow rebuild both. Benchmark data remains on
`gh-pages-rocjitsu`. See the
[handbook publishing setup](handbook/README.md#verification-and-publishing)
for the shared GitHub App and Pages configuration.
