# Rocjitsu web applications

This directory contains the public web applications for Rocjitsu.

| Directory | Purpose | Published path |
| --- | --- | --- |
| [dashboard/](dashboard/README.md) | Simulation-performance dashboard (React + Vite) | `/rocjitsu-dashboard/` |

Each application owns its dependencies, build configuration, and tests. See its
README for local development instructions. Canonical technical documentation
lives in [../docs/](../docs/).

The [publishing workflow](../../../.github/workflows/rocjitsu-publish-website.yml)
publishes the dashboard from `develop` to `gh-pages/rocjitsu-dashboard/`. Moving
the source directory does not change that public URL or the benchmark-data
location on `gh-pages-rocjitsu`.
