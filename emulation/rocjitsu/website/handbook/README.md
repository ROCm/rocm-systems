# Rocjitsu handbook

Publishing configuration for the Rocjitsu documentation and blog at
<https://rocm.github.io/rocm-systems/rocjitsu/>. The engineering-handbook
layout uses Material for MkDocs, system fonts, light/dark themes, search, and
Mermaid diagrams. The site uses a plain-text name without a logo.

Guide and blog Markdown lives in [../../docs/](../../docs/), the single source of
truth. MkDocs reads that directory directly: no copied guides, generated Markdown
mirror, or synchronization step. The dashboard introduction and blog
live there. The `docs/sphinx/` subtree is excluded from this publication.

This directory contains only publishing configuration, theme assets, and tests.
The homepage renders the existing [../../README.md](../../README.md) through an
in-memory MkDocs file. There is no separate `docs/index.md` or second overview.
Links between the README and published guides resolve to handbook pages,
including section anchors. README edits trigger CI and live reload.
The [simulation-performance dashboard](../dashboard/README.md) is a
separate application.

## Build and preview

Use Python 3.12. From this directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/mkdocs build --strict
.venv/bin/python -m unittest discover -s tests
node --test tests/repository-stats.test.cjs
.venv/bin/mkdocs serve --dev-addr 127.0.0.1:8000
```

The browser-script tests use Node.js 22.13 or newer with no npm dependencies.
Open the local address printed by MkDocs (including `/rocm-systems/rocjitsu/`).
The production build writes `site/`; generated files and the virtual environment
are ignored by Git. The pinned MkDocs 1.x and Material versions match the tested
theme and hook APIs; review compatibility before changing them.

## Editing

| Path | Purpose |
| --- | --- |
| `mkdocs.yml` | Navigation, theme, plugins, and production URL |
| `../../docs/` | Canonical guide pages, dashboard introduction, and blog |
| `../../README.md` | Homepage and project overview |
| `assets/handbook.css` | Layout, colors, and focus styles |
| `hooks.py` | Source-link resolution, Markdown renderers, and presentation icons |
| `overrides/partials/` | Repository link and counts |

Edit technical instructions in `../../docs/`; never add a second copy here.
The handbook presents complete documentation pages, not a directory of links to
raw Markdown. Keep layout, typography, and reading controls in this theme;
author page-specific structure once in the canonical Markdown. Material cards,
callouts, tabs, and expandable details are available where they aid reading.
They render on the website; GitHub's Markdown viewer may display their syntax
differently. Use tables for comparisons, not for homepage navigation.
The build adds guide-title icons without editing the canonical Markdown or its
heading anchors. Use plain text on subsection headings.
Sidebar icons belong to groups; individual navigation links remain text-only.
Keep explicit heading IDs stable when changing titles. The repository header
link points directly to Rocjitsu; its stars/forks describe the containing
`ROCm/rocm-systems` repository. A best-effort GitHub request at build time embeds
the counts in the HTML, so readers do not need API access. Browser requests
refresh them with a 15-minute session cache; failed refreshes retain existing
counts. Tooltips identify the initial build snapshot. If neither build nor
browser can fetch counts, the icons display an em dash. Set
`ROCJITSU_DOCS_OFFLINE=1` to skip the build request for offline builds.
The standard Material footer contains
the project copyright, license, GitHub and issue links, and theme attribution.
The README's GitHub CI badge is omitted during rendering, without changing its
Markdown source.

Relative links between documentation pages stay within the handbook. Links to
existing source files outside `docs/` open the corresponding GitHub location.
Missing targets still fail the strict build. Source files are never rewritten.

Blog posts live in `../../docs/blog/posts/` and use `date`, `title`, and `slug`
front matter. Keep unfinished articles marked `draft: true`; the production
build excludes them. To preview drafts locally:

```bash
.venv/bin/mkdocs serve --dev-addr 127.0.0.1:8000
```

Check technical claims against the corresponding document in `../../docs/` before
publishing a post. Diagrams must identify whether they show component organization
or a specific execution path.

## Verification and publishing

The `rocjitsu-handbook` CI workflow runs a strict build and regression tests
on Rocjitsu changes in pull requests and `develop` pushes, including source files
linked from the canonical docs.
Sphinx-only changes do not trigger it. Tests verify that canonical edits reach
the site, source links resolve, Sphinx is excluded, and builds do not write back
into `docs/`. Before submitting a change, run these checks and inspect affected
pages at desktop and mobile widths in both color themes.

The `rocjitsu-publish-website` workflow compares current `develop` with each site's
last published source revision, then builds and publishes only affected sites:

- Handbook: `rocjitsu/`
- Dashboard: `rocjitsu-dashboard/`

Changes anywhere in `emulation/rocjitsu/` select the handbook, since canonical
documentation can link to source files outside `docs/` and link resolution checks
whether those targets exist. Changes in `website/dashboard/` also select the
dashboard. Sphinx-only changes are excluded. A publishing-workflow change or a
manual run selects both.
Handbook-only runs do not install dashboard dependencies or update its Pages files.
Each published directory contains a `.source-revision` marker. Missing markers
trigger an initial build. Comparing against publication rather than the latest
push also catches changes from failed or superseded pending runs. Delayed runs
check out current `develop` so they cannot roll a site back to an older event.
If a history rewrite makes a recorded revision unavailable, a manual run rebuilds
both sites and refreshes their markers without using the old revisions.

Each selected site passes its verification suite before a publishing token is
created. The dashboard runs `npm run verify`, including production-hosting tests;
the handbook runs its strict build and Python/Node regression tests against the
same checkout that is published.

Both destinations use the same publishing workflow and concurrency group. Each
destination is replaced independently; files elsewhere on `gh-pages` are preserved.
Dashboard benchmark data remains on `gh-pages-rocjitsu`.

Publishing uses the existing `APP_ID` and `APP_PRIVATE_KEY` secrets to obtain a
GitHub App token with repository Contents write permission. The App must be
installed and allowed to push to `gh-pages`; Pages must use `gh-pages` at its
root. Manual publishing is restricted to `develop`. See the
[publishing workflow](../../../../.github/workflows/rocjitsu-publish-website.yml).

The root Pages index is maintained separately. This workflow does not overwrite
it.
